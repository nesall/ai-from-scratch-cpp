#pragma once
#include <vector>
#include <memory>
#include "matrix.hpp"
#include "optimizers.hpp"

namespace models {

  class LogisticRegression {
  private:
    std::vector<float> weights_;
    float bias_ = 0;
    float learning_rate_ = 0;
    int epochs_ = 0;
    std::unique_ptr<optimizers::Optimizer> opt_;

  public:
    LogisticRegression(float lr = 0.01f, int ep = 1000);
    void fit(const Matrix<float> &X, const std::vector<int> &y);
    int predict(std::span<const float> X) const;
    std::vector<int> predict(const Matrix<float> &X) const;
  };

}