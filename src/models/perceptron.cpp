#include "models/perceptron.hpp"
#include <cassert>
#include <random>

models::Perceptron::Perceptron(size_t inputs, float lr, size_t e)
{
  nofInputs_ = inputs;
  learningRate_ = lr;
  epochs_ = e;
  weights_.resize(nofInputs_ + 1, 0); // +1 for bias
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-0.5, 0.5);
  for (auto &w : weights_) {
    w = dis(gen);
  }
}

void models::Perceptron::fit(const std::vector<std::vector<float>> &X, const std::vector<int> &y)
{
  assert(X.size() == y.size() && "Input and target sizes must match.");
  assert(X[0].size() == nofInputs_ && "Input size must match the number of inputs.");
  for (size_t epoch = 0; epoch < epochs_; ++epoch) {
    for (size_t i = 0; i < X.size(); ++i) {
      float net_input = weights_[0]; // bias
      for (size_t j = 0; j < nofInputs_; ++j) {
        net_input += weights_[j + 1] * X[i][j];
      }
      int prediction = (net_input >= 0.0f) ? 1 : 0;
      int error = y[i] - prediction;
      weights_[0] += learningRate_ * error; // update bias
      for (size_t j = 0; j < nofInputs_; ++j) {
        weights_[j + 1] += learningRate_ * error * X[i][j];
      }
    }
  }
}

int models::Perceptron::predict(const std::vector<float> &x)
{
  assert(x.size() == nofInputs_ && "Input size must match the number of inputs.");
  float net_input = weights_[0]; // bias
  for (size_t j = 0; j < nofInputs_; ++j) {
    net_input += weights_[j + 1] * x[j];
  }
  return (net_input >= 0.0f) ? 1 : 0; // Activation function (step function)
}