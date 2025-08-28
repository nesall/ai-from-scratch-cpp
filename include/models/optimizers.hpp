#pragma once
#include <vector>
#include <cassert>
#include <iostream>
#include <utility>
#include <unordered_map>
#include <map>
#include <span>
#include <mutex>
#include <cmath>
#include <memory>
#include <numbers>
#include <optional>
#include <random>



namespace schedulers {

  class LRScheduler {
    mutable std::unordered_map<size_t, size_t> statsLrFreq_;
    mutable std::mutex mutex_;
  protected:
    std::unique_ptr<LRScheduler> inner_;
    void recordLr(float lr) const {
      std::unique_lock<std::mutex> lock(mutex_);
      statsLrFreq_[size_t(lr * 1'000'000.f)] ++;
      lock.unlock();

    }
  public:
    virtual ~LRScheduler() = default;
    virtual float get_lr(float base_lr, int step) const = 0;
    virtual void reset() {}  // For stateful schedulers

    void setInnerScheduler(std::unique_ptr<LRScheduler> p) {
      inner_ = std::move(p);
    }

    std::unordered_map<size_t, size_t> statsLrFreq() const { return statsLrFreq_; }
  };

  class ConstantLR : public LRScheduler {
  public:
    float get_lr(float base_lr, int step) const override {
      return base_lr;
    }
  };

  class WarmupCosineDecayLR : public LRScheduler {
  public:
    WarmupCosineDecayLR(int warmup_steps, int total_steps, float min_lr_ratio = 0.01f)
      : warmupSteps_(warmup_steps), totalSteps_(total_steps), minLrRatio_(min_lr_ratio) {
    }

    float get_lr(float base_lr, int step) const override {
      step = std::min(step, totalSteps_);
      float lr = base_lr;
      if (step < warmupSteps_) {
        // Linear warmup
        lr = base_lr * (static_cast<float>(step) / warmupSteps_);
      } else {
        // Cosine decay
        int decaySteps = step - warmupSteps_;
        int maxDecaySteps = totalSteps_ - warmupSteps_;
        float progress = std::min(static_cast<float>(decaySteps) / maxDecaySteps, 1.0f);
        float cosine_decay = 0.5f * (1.0f + std::cos(std::numbers::pi * progress));
        lr = base_lr * (minLrRatio_ + (1.0f - minLrRatio_) * cosine_decay);
      }
      if ((step % int(totalSteps_ / 100.f)) == 0) {
        //std::cout << "\tWarmupCosineDecayLR lr " << lr << " (step " << step << ")\n";
        recordLr(lr);
      }
      return lr;
    }

  private:
    int warmupSteps_;
    int totalSteps_;
    float minLrRatio_;
  };

  // Exponential Decay
  class ExponentialDecayLR : public LRScheduler {
  public:
    ExponentialDecayLR(float decay_rate = 0.95f, int decay_steps = 100)
      : decay_rate_(decay_rate), decay_steps_(decay_steps) {
    }

    float get_lr(float base_lr, int step) const override {
      return base_lr * std::pow(decay_rate_, step / decay_steps_);
    }

  private:
    float decay_rate_;
    int decay_steps_;
  };

  // Step Decay (reduce LR at specific milestones)
  class StepDecayLR : public LRScheduler {
  public:
    StepDecayLR(std::vector<int> milestones, float gamma = 0.1f)
      : milestones_(std::move(milestones)), gamma_(gamma) {
    }

    float get_lr(float base_lr, int step) const override {
      int reductions = 0;
      for (int milestone : milestones_) {
        if (step >= milestone) reductions++;
      }
      return base_lr * std::pow(gamma_, reductions);
    }

  private:
    std::vector<int> milestones_;
    float gamma_;
  };

  // Predefined set of lr that overwrites base_lr.
  class StepOverwriteLR : public LRScheduler {
  public:
    StepOverwriteLR(std::map<int, float> milestones) : milestones_(std::move(milestones)) {}
    float get_lr(float base_lr, int step) const override {
      for (auto a : milestones_) {
        if (a.first <= step) base_lr = a.second;
      }
      return base_lr;
    }
  private:
    std::map<int, float> milestones_;
  };

  class NoiseInducingLR : public LRScheduler {
  public:
    enum class DistributionType { Uniform, Normal };
    enum class NoiseDecayType { Linear, Cosine, Exponential };
    NoiseInducingLR(float noiseRatio = 0.02f,
      std::optional<int> seed = std::nullopt,
      std::optional<size_t> totalSteps = std::nullopt,
      DistributionType distrType = DistributionType::Uniform,
      NoiseDecayType decayType = NoiseDecayType::Linear)
      : noiseRatio_(noiseRatio),
      rng_(seed.has_value() ? seed.value() : int(std::random_device{}())),
      totalSteps_(totalSteps),
      distrType_(distrType),
      decayType_(decayType)
    {
    }
    float get_lr(float base_lr, int step) const override {
      if (inner_) base_lr = inner_->get_lr(base_lr, step);
      float decayFactor = 1.0f;
      if (totalSteps_.has_value()) {
        float progress = static_cast<float>(step) / *totalSteps_;
        switch (decayType_) {
        case NoiseDecayType::Linear:
          decayFactor = 1.0f - progress;
          break;
        case NoiseDecayType::Cosine:
          decayFactor = 0.5f * (1.0f + std::cos(std::numbers::pi * progress));
          break;
        case NoiseDecayType::Exponential:
          decayFactor = std::exp(-5.0f * progress);  // Tunable decay rate
          break;
        }
      }
      float noise = 0.0f;
      switch (distrType_) {
      case DistributionType::Uniform:
        if (1) {
          std::uniform_real_distribution<float> dist(-base_lr * noiseRatio_, base_lr * noiseRatio_);
          noise = dist(rng_);
        }
        break;
      case DistributionType::Normal:
        if (0 < base_lr) {
          std::normal_distribution<float> dist(0.0f, base_lr * noiseRatio_ / 2.0f);  // stddev scaled
          noise = dist(rng_);
        }
        break;
      }
      const auto lr = base_lr + noise * decayFactor;
      if (totalSteps_.has_value() && (step % int(*totalSteps_ / 100.f)) == 0) {
        //std::cout << "\tWarmupCosineDecayLR lr " << lr << " (step " << step << ")\n";
        recordLr(lr);
      }
      return lr;
    }
  private:
    float noiseRatio_;
    mutable std::mt19937 rng_;
    std::optional<size_t> totalSteps_;
    NoiseDecayType decayType_;
    DistributionType distrType_;
  };

  // Cyclical Learning Rate (for finding good learning rates)
  class CyclicalLR : public LRScheduler {
  public:
    CyclicalLR(float min_lr, float max_lr, int cycle_length)
      : min_lr_(min_lr), max_lr_(max_lr), cycle_length_(cycle_length) {
    }

    float get_lr(float base_lr, int step) const override {
      int cycle_pos = step % cycle_length_;
      float progress = static_cast<float>(cycle_pos) / cycle_length_;
      return min_lr_ + (max_lr_ - min_lr_) * (0.5f * (1.0f + std::sin(2 * std::numbers::pi * progress - std::numbers::pi / 2)));
    }

  private:
    float min_lr_, max_lr_;
    int cycle_length_;
  };

  // Advanced - for escaping local minimas
  class AdaptiveRestartLR : public LRScheduler {
  public:
    AdaptiveRestartLR(float initial_lr, int restart_period = 200, float restart_factor = 0.8f)
      : initial_lr_(initial_lr), restart_period_(restart_period), restart_factor_(restart_factor) {
    }

    float get_lr(float base_lr, int step) const override {
      int cycle = step / restart_period_;
      int cycle_step = step % restart_period_;

      // Cosine decay within each cycle
      float cycle_lr = initial_lr_ * std::pow(restart_factor_, cycle);
      float progress = static_cast<float>(cycle_step) / restart_period_;
      float cosine_factor = 0.5f * (1.0f + std::cos(std::numbers::pi * progress));

      return cycle_lr * cosine_factor;
    }

  private:
    float initial_lr_;
    int restart_period_;
    float restart_factor_;
  };

} // namespace schedulers


//----------------------------------------------------------------------------------------------


namespace optimizers {

  class Optimizer {
  public:
    Optimizer() {
      resetScheduler(std::make_unique<schedulers::ConstantLR>());
    }
    virtual ~Optimizer() = default;
    Optimizer(const Optimizer &) = delete;
    Optimizer &operator =(const Optimizer &) = delete;

    virtual void update(std::span<float> params, std::span<const float> grads, int context = -1) = 0;
    void update(std::vector<float> &params, const std::vector<float> &grads, int context = -1) {
      update(std::span<float>(params), std::span<const float>(grads));
    }
    void update(float &w, float grad) {
      float param_arr[1] = { w };
      float grad_arr[1] = { grad };
      update(std::span<float>(param_arr, 1), std::span<const float>(grad_arr, 1));
      w = param_arr[0];
    }

    void resetScheduler(std::unique_ptr<schedulers::LRScheduler> p) {
      scheduler_ = std::move(p);
    }
    void incrementStep() { global_step_++; }

  protected:
    // Helper for derived classes to get scheduled learning rate
    float get_scheduled_lr(float base_lr) {
      current_lr_ = scheduler_ ? scheduler_->get_lr(base_lr, global_step_) : base_lr;
      return current_lr_;
    }

  private:
    std::unique_ptr<schedulers::LRScheduler> scheduler_;
    int global_step_ = 0;
    float current_lr_ = 0.0f;
  };


  // SGD optimizer refers to the bare-bones gradient descent stepper (batch, stochastic, or mini - batch).
  // It’s more like the default baseline optimizer, and everything else is an enhancement on top of it.
  // Models: Linear, Logistic, SVM, Perceptron
  class SGD : public Optimizer {
  public:
    SGD(float lr) : lr_(lr) {}

    void update(std::span<float> params, std::span<const float> grads, int context = -1) override {
      assert(params.size() == grads.size());
      float lr = get_scheduled_lr(lr_);
      for (size_t i = 0; i < params.size(); ++i) {
        params[i] -= lr * grads[i];
      }
      incrementStep();
    }

  private:
    float lr_ = 0.f;
  };


  // Models: Linear, Logistic, SVM, Perceptron
  class Momentum : public Optimizer {
  public:
    Momentum(float lr, float beta = 0.9f) : lr_(lr), beta_(beta) {}

    void update(std::span<float> params, std::span<const float> grads, int context = -1) override {
      assert(params.size() == grads.size());
      std::unique_lock<std::mutex> lock(mutex_);
      auto &vv = velocity_[context];
      if (vv.size() != params.size()) {
        vv.resize(params.size(), 0.0f);
      }
      lock.unlock();
      float lr = get_scheduled_lr(lr_);
      for (size_t i = 0; i < params.size(); ++i) {
        vv[i] = beta_ * vv[i] - lr * grads[i];
        params[i] += vv[i];
      }
      incrementStep();
    }

  private:
    float lr_;
    float beta_;
    std::unordered_map<int, std::vector<float>> velocity_;
    std::mutex mutex_;
  };


  // Models: MLP, CNN, RNN
  class RMSProp : public Optimizer {
  public:
    RMSProp(float lr, float beta = 0.9f, float epsilon = 1e-8f) : lr_(lr), beta_(beta), epsilon_(epsilon) {}
 
    void update(std::span<float> params, std::span<const float> grads, int context = -1) override {
      assert(params.size() == grads.size());
      std::unique_lock<std::mutex> lock(mutex_);
      auto &avg_grads = avg_sq_grad_[context];
      if (avg_grads.size() != params.size()) {
        avg_grads.resize(params.size(), 0.0f);
      }
      lock.unlock();
      float lr = get_scheduled_lr(lr_);
      for (size_t i = 0; i < params.size(); ++i) {
        avg_grads[i] = beta_ * avg_grads[i] + (1 - beta_) * grads[i] * grads[i];
        params[i] -= lr * grads[i] / (std::sqrt(avg_grads[i]) + epsilon_);
      }
      incrementStep();
    }

  private:
    float lr_;
    float beta_;
    float epsilon_;
    std::unordered_map<int, std::vector<float>> avg_sq_grad_;  // moving average of squared gradients
    std::mutex mutex_;
  };


  // Models: MLP, CNN, RNN
  class Adam : public Optimizer {
  public:
    Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f) 
      : lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {
    }

    void update(std::span<float> params, std::span<const float> grads, int context = -1) override {
      assert(params.size() == grads.size());
      std::unique_lock<std::mutex> lock(mutex_);
      auto &ctx = contexts_[context];
      if (ctx.m_.size() != params.size()) {
        ctx.m_.resize(params.size(), 0.0f);
        ctx.v_.resize(params.size(), 0.0f);
      }
      ctx.t_ += 1;
      lock.unlock();
      const float beta1_t = std::pow(beta1_, ctx.t_);
      const float beta2_t = std::pow(beta2_, ctx.t_);
      const float lr = get_scheduled_lr(lr_);
      for (size_t i = 0; i < params.size(); ++i) {
        const auto grad = std::clamp(grads[i], -1.f, 1.f);
        // Update biased moment estimates
        ctx.m_[i] = beta1_ * ctx.m_[i] + (1 - beta1_) * grad;
        ctx.v_[i] = beta2_ * ctx.v_[i] + (1 - beta2_) * grad * grad;
        // Compute bias-corrected moment estimates
        float m_hat = ctx.m_[i] / (1.0f - beta1_t);
        float v_hat = ctx.v_[i] / (1.0f - beta2_t);
        // Update parameters
        params[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon_);
      }
      //incrementStep();
    }

  private:
    float lr_;
    float beta1_;
    float beta2_;
    float epsilon_;
    struct ContextData {
      std::vector<float> m_; // first moment (mean of gradients)
      std::vector<float> v_; // second moment (uncentered variance)
      int t_ = 0; // timestep
    };
    std::unordered_map<int, ContextData> contexts_;  // moving average of squared gradients
    std::mutex mutex_;
  };

}