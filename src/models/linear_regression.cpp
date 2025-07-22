#include "../include/models/linear_regression.hpp"
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <limits>

using namespace models;

LinearRegression::LinearRegression(float lr, int ep)
{
  learning_rate_ = lr;
  epochs_ = ep;
  if (learning_rate_ > 1.0f) {
    std::cerr << "LinearRegression - Warning: High learning rate (" << learning_rate_ << ") may cause divergence.\n";
  }
}

void LinearRegression::fit(const std::vector<float> &x, const std::vector<float> &y)
{
  if (x.size() != y.size() || x.empty()) {
    throw std::invalid_argument("Input vectors must be of the same non-zero length.");
  }
  weight_ = 0.0f;
  bias_ = 0.0f;
  float last_mse = std::numeric_limits<float>::max();
  // Gradient descent
  for (int epoch = 0; epoch < epochs_; ++epoch) {
    float weight_gradient = 0.0f;
    float bias_gradient = 0.0f;
    float mse = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
      float prediction = weight_ * x[i] + bias_;
      float error = prediction - y[i];
      weight_gradient += error * x[i];
      bias_gradient += error;
      mse += error * error;
    }
    weight_ -= learning_rate_ * (weight_gradient / x.size());
    bias_ -= learning_rate_ * (bias_gradient / x.size());
    mse /= x.size();
    if (std::fabs(last_mse - mse) < 1e-6f) {
      std::cout << "LinearRegression - Early stopping at epoch " << epoch << " with MSE: " << mse << '\n';
      break;
    }
    last_mse = mse;
  }
}

float LinearRegression::predict(float x)
{
  if (weight_ == 0.0f && bias_ == 0.0f) {
    throw std::runtime_error("Model has not been trained yet.");
  }
  return weight_ * x + bias_;
}

std::vector<float> LinearRegression::predict(const std::vector<float> &x_vals)
{
  std::vector<float> res;
  std::transform(x_vals.cbegin(), x_vals.cend(), std::back_inserter(res), [this](auto x) { return predict(x); });
  return res;
}
