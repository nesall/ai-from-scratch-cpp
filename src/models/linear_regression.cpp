#include "models/linear_regression.hpp"
#include <cassert>
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
  opt_ = std::make_unique<optimizers::SGD>(lr);
}

void LinearRegression::fit(const Matrix<float> &X, const std::vector<float> &y)
{
  assert(!X.empty() && "Input vector must be non-zero length.");
  assert(X.rows() == y.size() && "Feature matrix and target vector must have same number of samples.");
  const auto nofSamples = X.rows();
  const auto nofFeatures = X.cols();
  WeightInitializer::initialize_vector(weights_, nofFeatures, Initialization::RandomUniform);
  bias_ = 0.0f;
  float last_mse = std::numeric_limits<float>::max();
  // Butch Gradient Descent
  for (int epoch = 0; epoch < epochs_; ++epoch) {
    std::vector<float> weight_gradients(nofFeatures, 0.0f);
    float bias_gradient = 0.0f;
    float mse = 0.0f;
    for (size_t i = 0; i < nofSamples; i ++) {
      float prediction = bias_;
      for (size_t j = 0; j < nofFeatures; j ++) {
        prediction += weights_[j] * X(i, j);
      }
      float error = prediction - y[i];
      for (size_t j = 0; j < nofFeatures; ++j) {
        weight_gradients[j] += error * X(i, j) / nofSamples;
      }
      bias_gradient += error / nofSamples;
      mse += error * error;
    }
    for (size_t j = 0; j < nofFeatures; j ++) {
      //weights_[j] -= learning_rate_ * weight_gradients[j];
      opt_->update(weights_, weight_gradients);
    }
    //bias_ -= learning_rate_ * (bias_gradient / nofSamples);
    opt_->update(bias_, bias_gradient);
    if (stepf_ && !stepf_()) {
      break;
    }
    mse /= nofSamples;
    if (std::fabs(last_mse - mse) < 1e-6f) {
      std::cout << "\tearly stoppage, epoch " << epoch << ", mse " << mse << '\n';
      break;
    }
    last_mse = mse;
  }
}

float LinearRegression::predict(std::span<const float> X)
{
  assert (!weights_.empty() && "Model has not been trained yet."); 
  //return weight_ * x + bias_;
  assert(X.size() == weights_.size());
  float result = bias_;
  for (size_t i = 0; i < X.size(); ++i) {
    result += weights_[i] * X[i];
  }
  return result;
}

std::vector<float> LinearRegression::predict(const Matrix<float> &X)
{
  std::vector<float> res;
  for (auto row : X) {
    //std::transform(row.begin(), row.end(), std::back_inserter(res), [this](auto x) { return predict(x); });
    res.push_back(predict(row));
  }
  return res;
}
