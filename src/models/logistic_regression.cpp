#include "models/logistic_regression.hpp"
#include "common.hpp"
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <limits>

using namespace models;

LogisticRegression::LogisticRegression(float lr, int ep)
{
  learning_rate_ = lr;
  epochs_ = ep;
  if (learning_rate_ > 1.0f) {
    std::cerr << "LogisticRegression - Warning: High learning rate (" << learning_rate_ << ") may cause divergence.\n";
  }
  opt_ = std::make_unique<optimizers::SGD>(lr);
}


void LogisticRegression::fit(const Matrix<float> &X, const std::vector<int> &y)
{
  assert(!X.empty() && "Input vector must be non-zero length.");
  assert(X.rows() == y.size() && "Feature matrix and target vector must have same number of samples.");
  const auto nofSamples = X.rows();
  const auto nofFeatures = X.cols();
  WeightInitializer::initialize_vector(weights_, nofFeatures, Initialization::RandomUniform);
  bias_ = 0.0f;
  float last_mse = std::numeric_limits<float>::max();
  for (int epoch = 0; epoch < epochs_; ++epoch) {
    std::vector<float> weight_gradients(nofFeatures, 0.0f);
    float bias_gradient = 0.0f;
    float total_loss = 0.0f;
    for (size_t i = 0; i < nofSamples; ++i) {
      float z = bias_;
      for (size_t j = 0; j < nofFeatures; j ++) {
        z += weights_[j] * X(i, j);
      }
      float prediction = utils::sigmoid(z);
      float error = prediction - y[i];
      // Update weights and bias
      for (size_t j = 0; j < nofFeatures; ++j) {
        weight_gradients[j] += error * X(i, j) / nofSamples;
      }
      bias_gradient += error / nofSamples;

      constexpr float eps = 1e-15f;
      float clamped_pred = std::clamp(prediction, eps, 1.0f - eps); // for numerical stability
      total_loss += -y[i] * std::log(clamped_pred) - (1 - y[i]) * std::log(1 - clamped_pred);
    }
    
    for (size_t j = 0; j < nofFeatures; j ++) {
      opt_->update(weights_, weight_gradients);
    }
    opt_->update(bias_, bias_gradient);
    // Optionally print loss for monitoring
    if (epoch % 100 == 0) {
      //std::cout << "Epoch " << epoch << ", Loss: " << total_loss / x.size() << '\n';
    }
    total_loss /= nofSamples;
    if (std::fabs(last_mse - total_loss) < 1e-6f) {
      std::cout << "\tearly stoppage, epoch " << epoch << ", mse " << total_loss << '\n';
      break;
    }
    last_mse = total_loss;
  }
}

int LogisticRegression::predict(std::span<const float> X) const
{
  assert(!X.empty());
  assert(X.size() == weights_.size());

  float z = bias_;
  for (size_t i = 0; i < X.size(); ++i) {
    z += weights_[i] * X[i];
  }
  
  float prob = utils::sigmoid(z);

  if (prob >= 0.5f) {
    return 1; // Class 1
  } else {
    return 0; // Class 0
  }
}

std::vector<int> LogisticRegression::predict(const Matrix<float> &x_vals) const
{
  assert(!weights_.empty() && "Model has not been trained yet.");
  assert(!x_vals.empty());
  std::vector<int> predictions;
  predictions.reserve(x_vals.rows());
  for (const auto &x : x_vals) {
    predictions.push_back(predict(x));
  }
  return predictions;
}
