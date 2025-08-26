#pragma once

#include <vector>
#include <random>
#include <span>
#include <cassert>

namespace regularization {

  class RegBase {
  public:
    virtual void apply(std::span<float> params) = 0;
    void apply(std::vector<float> &params) { apply(std::span<float>(params)); }
  };

  class L1 : public RegBase {
  public:
    L1(float lambda = 0.001f) : lambda_(lambda) {}
    void apply(std::span<float> params) override {
      for (auto &p : params) {
        p -= lambda_ * (0 < p ? 1.0f : -1.0f); // shrink towards 0
      }
    }
  private:
    float lambda_;
  };

  class L2 : public RegBase {
  public:
    L2(float lambda = 0.001f) : lambda_(lambda) {}
    void apply(std::span<float> params) override {
      for (auto &p : params) {
        p -= lambda_ * p; // shrink smoothly towards 0
      }
    }
  private:
    float lambda_;
  };

  class ElasticNet : public RegBase {
  public:
    ElasticNet(float l1 = 0.001f, float l2 = 0.001f) : l1_(l1), l2_(l2) {}
    void apply(std::span<float> params) override {
      for (auto &p : params) {
        p -= l1_ * (0 < p ? 1.0f : -1.0f); // L1
        p -= l2_ * p;                      // L2
      }
    }
  private:
    float l1_, l2_;
  };


  class Dropout {
  public:
    Dropout(float rate) : rate_(rate), gen_(rd_()), dist_(0.0f, 1.0f) {
      assert(rate_ >= 0.0f && rate_ < 1.0f);
    }

    void apply(std::span<float> activations, bool training = true) {
      if (!training) {
        // Scale outputs during inference
        for (auto &a : activations) {
          a *= (1.0f - rate_);
        }
        return;
      }

      mask_.resize(activations.size());
      for (size_t i = 0; i < activations.size(); ++i) {
        mask_[i] = (dist_(gen_) >= rate_) ? 1.0f : 0.0f;
        activations[i] *= mask_[i];
      }
    }
    void apply(std::vector<float> &activations, bool training = true) {
      apply(std::span<float>(activations), training);
    }

  private:
    float rate_;
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_real_distribution<float> dist_;
    std::vector<float> mask_;
  };


  class EarlyStopping {
  public:
    EarlyStopping(int patience = 20)
      : patience_(patience), bestLoss_(std::numeric_limits<float>::max()), counter_(0), stop_(false) {
    }

    bool update(float loss) {
      if (stopping()) return true;
      if (loss < bestLoss_) {
        bestLoss_ = loss;
        counter_ = 0;
      } else if (!isnan(loss)) {
        counter_++;
        if (counter_ >= patience_) {
          stop_ = true;
        }
      }
      return stopping();
    }

    bool stopping() const { return stop_; }

  private:
    int patience_;
    float bestLoss_;
    int counter_;
    bool stop_;
  };

} // namespace regularization
