#pragma once
#include <vector>
#include <memory>
#include "matrix.hpp"
#include "common.hpp"
#include "optimizers.hpp"

namespace models {

  class LinearRegression : public utils::StepBreaker {
  private:
    std::vector<float> weights_;
    float bias_ = 0;
    float learning_rate_ = 0;
    int epochs_ = 0;
    std::unique_ptr<optimizers::Optimizer> opt_;

  public:
    LinearRegression(float lr = 0.01f, int ep = 1000);

    void fit(const Matrix<float> &X, const std::vector<float> &y);
    float predict(std::span<const float> X);
    std::vector<float> predict(const Matrix<float> &X);

  private:
    LinearRegression(const LinearRegression &) = delete;
    LinearRegression &operator =(const LinearRegression &) = delete;
  };

}
