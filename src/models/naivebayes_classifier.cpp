#include "models/naivebayes_classifier.hpp"
#include <algorithm>
#include <stdexcept>
#include <iterator>
#include <numbers>
#include <cassert>

using namespace models;

void GaussianNaiveBayes::fit(const std::vector<float> &x_train, const std::vector<int> &y_train)
{
  if (x_train.empty() || x_train.size() != y_train.size()) {
    throw std::invalid_argument("x_train and y_train must have the same length");
  }
  model_.clear();
  const size_t nSamples = x_train.size();
  for (size_t i = 0; i < nSamples; ++i) {
    int label = y_train[i];
    float x = x_train[i];
    auto &stats = model_[label];
    stats.prior += 1.f / float(nSamples);
    stats.mean += x;
    stats.variance += x * x;
    stats.nof ++;
  }
  for (auto &pair : model_) {
    int label = pair.first;
    ClassStats &stats = pair.second;
    stats.mean /= float(stats.nof);
    stats.variance = (stats.variance / stats.nof) - (stats.mean * stats.mean);  // Calculate variance
    assert(0 <= stats.variance);
  }
}

int GaussianNaiveBayes::predict(float x_test)
{
  if (model_.empty()) {
    throw std::runtime_error("Model has not been trained yet");
  }
  float max_prob = -1.0f;
  int best_label = -1;
  for (const auto &pair : model_) {
    int label = pair.first;
    const ClassStats &stats = pair.second;
    // Calculate Gaussian probability density function
    float variance = stats.variance;
    if (variance == 0) {
      continue;  // Skip if variance is zero to avoid division by zero
    }
    float exponent = -((x_test - stats.mean) * (x_test - stats.mean)) / (2 * variance);
    float prob = (1.0f / std::sqrt(2 * std::numbers::pi * variance)) * std::exp(exponent);
    prob *= stats.prior;  // Incorporate prior probability
    if (prob > max_prob) {
      max_prob = prob;
      best_label = label;
    }
  }
  return best_label;
}

std::vector<int> GaussianNaiveBayes::predict(const std::vector<float> &x_tests)
{
  std::vector<int> res;
  std::transform(x_tests.cbegin(), x_tests.cend(), std::back_inserter(res), [this](float x) { return predict(x); });
  return res;
}