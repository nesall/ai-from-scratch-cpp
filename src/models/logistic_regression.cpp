#include "models/logistic_regression.hpp"
#include <stdexcept>
#include <algorithm>
#include <iterator>
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
}


void LogisticRegression::fit(const std::vector<float> &x, const std::vector<int> &y)
{
  if (x.size() != y.size()) {
    throw std::invalid_argument("Input x and labels y must have the same length.");
  }
  if (x.empty()) {
    throw std::invalid_argument("Input x cannot be empty.");
  }
  // Initialize weights and bias
  weight_ = 0.0f; // Could be initialized randomly
  bias_ = 0.0f;
  for (int epoch = 0; epoch < epochs_; ++epoch) {
    float total_loss = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
      float z = weight_ * x[i] + bias_;
      float prediction = sigmoid(z);
      float error = prediction - y[i];
      // Update weights and bias
      weight_ -= learning_rate_ * error * x[i];
      bias_ -= learning_rate_ * error;
      total_loss += -y[i] * std::log(prediction) - (1 - y[i]) * std::log(1 - prediction);
    }
    // Optionally print loss for monitoring
    if (epoch % 100 == 0) {
      //std::cout << "Epoch " << epoch << ", Loss: " << total_loss / x.size() << '\n';
    }
  }
}

float LogisticRegression::predict_proba(float x) const
{
  if (x < 0) {
    throw std::invalid_argument("Input x must be non-negative for logistic regression.");
  }
  float z = weight_ * x + bias_;
  float prob = sigmoid(z);
  return prob;
}

int LogisticRegression::predict(float x) const
{
  if (x < 0) {
    throw std::invalid_argument("Input x must be non-negative for logistic regression.");
  }
  float prob = predict_proba(x);
  if (prob >= 0.5f) {
    return 1; // Class 1
  } else {
    return 0; // Class 0
  }
}

std::vector<int> LogisticRegression::predict(const std::vector<float> &x_vals) const
{
  if (x_vals.empty()) {
    throw std::invalid_argument("Input x_vals cannot be empty.");
  }
  std::vector<int> predictions;
  predictions.reserve(x_vals.size());
  for (const auto &x : x_vals) {
    if (x < 0) {
      throw std::invalid_argument("Input x must be non-negative for logistic regression.");
    }
    predictions.push_back(predict(x));
  }
  return predictions;
}

float LogisticRegression::sigmoid(float z) const
{
  if (z >= 0) {
    float exp_neg_z = std::exp(-z);
    return 1.0f / (1.0f + exp_neg_z);
  } else {
    float exp_z = std::exp(z);
    return exp_z / (1.0f + exp_z);
  }  
}
