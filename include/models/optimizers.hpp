#pragma once
#include <vector>
#include <cassert>
#include <unordered_map>
#include <span>
#include <mutex>
#include <cmath>
#include <memory>
#include <numbers>


namespace optimizers {

  class LRScheduler {
  public:
    virtual ~LRScheduler() = default;
    virtual float get_lr(float base_lr, int step) const = 0;
    virtual void reset() {}  // For stateful schedulers
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
      : warmup_steps_(warmup_steps), total_steps_(total_steps), min_lr_ratio_(min_lr_ratio) {
    }

    float get_lr(float base_lr, int step) const override {
      if (step < warmup_steps_) {
        // Linear warmup
        return base_lr * (static_cast<float>(step) / warmup_steps_);
      } else {
        // Cosine decay
        int decay_steps = step - warmup_steps_;
        int max_decay_steps = total_steps_ - warmup_steps_;
        float progress = std::min(static_cast<float>(decay_steps) / max_decay_steps, 1.0f);
        float cosine_decay = 0.5f * (1.0f + std::cos(std::numbers::pi * progress));
        return base_lr * (min_lr_ratio_ + (1.0f - min_lr_ratio_) * cosine_decay);
      }
    }

  private:
    int warmup_steps_;
    int total_steps_;
    float min_lr_ratio_;
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
  class AdaptiveRestartLR : public optimizers::LRScheduler {
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


  //----------------------------------------------------------------------------------------------


  class Optimizer {
  public:
    Optimizer() {
      reset_scheduler(std::make_unique<ConstantLR>());
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

    void reset_scheduler(std::unique_ptr<LRScheduler> p) {
      scheduler_ = std::move(p);
    }

  protected:
    // Helper for derived classes to get scheduled learning rate
    float get_scheduled_lr(float base_lr) {
      current_lr_ = scheduler_ ? scheduler_->get_lr(base_lr, global_step_) : base_lr;
      return current_lr_;
    }
    void increment_step() { global_step_++; }

  private:
    std::unique_ptr<LRScheduler> scheduler_;
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
      increment_step();
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
      if (velocity_.size() != params.size()) {
        velocity_.resize(params.size(), 0.0f);
      }
      float lr = get_scheduled_lr(lr_);
      for (size_t i = 0; i < params.size(); ++i) {
        velocity_[i] = beta_ * velocity_[i] - lr * grads[i];
        params[i] += velocity_[i];
      }
      increment_step();
    }

  private:
    float lr_;
    float beta_;
    std::vector<float> velocity_;
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
      increment_step();
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
      float lr = get_scheduled_lr(lr_);
      for (size_t i = 0; i < params.size(); ++i) {
        // Update biased moment estimates
        ctx.m_[i] = beta1_ * ctx.m_[i] + (1 - beta1_) * grads[i];
        ctx.v_[i] = beta2_ * ctx.v_[i] + (1 - beta2_) * grads[i] * grads[i];
        // Compute bias-corrected moment estimates
        float m_hat = ctx.m_[i] / (1.0f - beta1_t);
        float v_hat = ctx.v_[i] / (1.0f - beta2_t);
        // Update parameters
        params[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon_);
      }
      increment_step();
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