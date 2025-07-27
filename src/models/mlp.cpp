#include "models/mlp.hpp"
#include "common.hpp"
#include <random>
#include <cassert>

models::MLP::MLP(const std::vector<size_t> &layers, float learning_rate)
{
  assert(layers.size() >= 2 && "Need at least input and output layer");
  layerSizes_ = layers;
  learningRate_ = learning_rate;
  layers_.resize(layers.size() - 1); // omit input layer (as it has no weights)
  for (size_t i = 0; i < layers.size() - 1; ++i) {
    const int currentLayerSize = layerSizes_[i + 1];  // neurons in current layer
    const int prevLayerSize = layerSizes_[i];         // inputs from previous layer
    assert(currentLayerSize > 0 && "Layer size must be greater than 0");
    Layer layer;
    layer.weights.resize(currentLayerSize, std::vector<float>(prevLayerSize));
    layer.biases.resize(currentLayerSize, 0.0f);
    layer.activations.resize(currentLayerSize, 0.0f);
    layer.deltas.resize(currentLayerSize, 0.0f);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);
    for (auto &weights : layer.weights) {
      for (auto &w : weights) {
        w = dis(gen);
      }
    }
    layers_[i] = std::move(layer);
  }
}

void models::MLP::fit(const std::vector<std::vector<float>> &X, const std::vector<std::vector<float>> &y, size_t epochs)
{
  assert(X.size() == y.size() && "Input and target sizes must match");
  assert(!X.empty() && X[0].size() == layerSizes_[0] && "Input size must match first layer size");
  assert(layerSizes_.back() == y[0].size() && "Output size must match last layer size");
  assert(layerSizes_.back() == 1 && "Currently only supports single output for binary classification");
  for (int epoch = 0; epoch < epochs; ++epoch) {
    for (size_t i = 0; i < X.size(); ++i) {
      std::vector<int> target(layerSizes_.back());
      target[0] = static_cast<int>(y[i][0]);  // Assuming single output for binary classification
      forwardPass(X[i]);
      backwardPass(X[i], target);
      updateWeights(X[i]);
    }
  }
}

std::vector<float> models::MLP::predict(const std::vector<float> &x)
{
  assert(x.size() == layerSizes_[0] && "Input size must match the input layer size");
  forwardPass(x); // Perform forward pass to compute activations
  return layers_.back().activations;
}

void models::MLP::forwardPass(const std::vector<float> &input)
{
  assert(input.size() == layerSizes_[0] && "Input size must match the input layer size");
  auto currentInput = input;
  for (size_t i = 0; i < layers_.size(); ++i) {
    for (size_t j = 0; j < layers_[i].weights.size(); ++j) {
      float netInput = layers_[i].biases[j];
      for (size_t k = 0; k < currentInput.size(); ++k) {
        netInput += layers_[i].weights[j][k] * currentInput[k];
      }
      layers_[i].activations[j] = commons::sigmoid(netInput);
    }
    currentInput = layers_[i].activations; // Pass activations
  }
}

void models::MLP::backwardPass(const std::vector<float> &input, const std::vector<int> &target) {
  assert(target.size() == layers_.back().activations.size() && "Target size must match output layer size");
  // 1. Calculate output layer deltas (error * derivative)
  size_t outputLayerIdx = layers_.size() - 1;
  for (size_t j = 0; j < layers_[outputLayerIdx].activations.size(); ++j) {
    float output = layers_[outputLayerIdx].activations[j];
    float error = target[j] - output;
    layers_[outputLayerIdx].deltas[j] = error * commons::sigmoidDerivative(output);
  }
  // 2. Calculate hidden layer deltas (propagate error backwards)
  for (int i = layers_.size() - 2; i >= 0; --i) {
    for (size_t j = 0; j < layers_[i].activations.size(); ++j) {
      float error = 0.0f;
      for (size_t k = 0; k < layers_[i + 1].deltas.size(); ++k) {
        error += layers_[i + 1].deltas[k] * layers_[i + 1].weights[k][j];
      }
      layers_[i].deltas[j] = error * commons::sigmoidDerivative(layers_[i].activations[j]);
    }
  }
}

void models::MLP::updateWeights(const std::vector<float> &input)
{
  for (size_t i = 0; i < layers_.size(); ++i) {
    std::vector<float> layerInput;
    if (i == 0) {
      layerInput = input;
    } else {
      layerInput = layers_[i - 1].activations;
    }
    // Update weights and biases for each neuron in this layer
    for (size_t j = 0; j < layers_[i].weights.size(); ++j) {
      layers_[i].biases[j] += learningRate_ * layers_[i].deltas[j];
      for (size_t k = 0; k < layers_[i].weights[j].size(); ++k) {
        layers_[i].weights[j][k] += learningRate_ * layers_[i].deltas[j] * layerInput[k];
      }
    }
  }
}
