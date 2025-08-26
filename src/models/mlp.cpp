#include "models/mlp.hpp"
#include "common.hpp"
#include "models/regularization.hpp"
#include <random>
#include <cassert>
#include <thread>
#include <execution>
#include <numeric>
#include <iostream>
#include <algorithm>

#define USE_PARALLEL 1

models::MLP::MLP(const std::vector<size_t> &layers, float learning_rate, Initialization ini)
{
  assert(2 <= layers.size() && "Need at least input and output layer");
  layerSizes_ = layers;
  learningRate_ = learning_rate;
  layers_.resize(layers.size() - 1); // omit input layer (as it has no weights)
  for (size_t i = 0; i < layers.size() - 1; ++i) {
    const int currentLayerSize = layerSizes_[i + 1];  // neurons in current layer
    const int prevLayerSize = layerSizes_[i];         // inputs from previous layer
    assert(0 < currentLayerSize && "Layer size must be greater than 0");
    Layer layer;
    layer.weights.resize(currentLayerSize, prevLayerSize);
    layer.biases.resize(currentLayerSize, 0.0f);
    layer.activations.resize(currentLayerSize, 0.0f);
    layer.deltas.resize(currentLayerSize, 0.0f);
    layers_[i] = std::move(layer);
    WeightInitializer::initialize_matrix(layers_[i].weights, currentLayerSize, prevLayerSize, ini);
  }
  opt_ = std::make_unique<optimizers::SGD>(learning_rate);
}

void models::MLP::fit(const std::vector<std::vector<float>> &X, const std::vector<std::vector<float>> &y, 
  size_t epochs, ActivationF af, float validationRatio, size_t patience)
{
  assert(X.size() == y.size() && "Input and target sizes must match");
  assert(!X.empty() && X[0].size() == layerSizes_[0] && "Input size must match first layer size");
  assert(layerSizes_.back() == y[0].size() && "Output size must match last layer size");
  af_ = af;

  // Shuffle indices
  std::vector<size_t> indices(X.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices.begin(), indices.end(), g);

  const size_t val_size = static_cast<size_t>(X.size() * validationRatio);
  const size_t train_size = X.size() - val_size;
  std::vector<std::vector<float>> X_train, y_train, X_val, y_val;
  X_train.reserve(train_size);
  y_train.reserve(train_size);
  X_val.reserve(val_size);
  y_val.reserve(val_size);
  for (size_t i = 0; i < train_size; ++i) {
    X_train.push_back(X[indices[i]]);
    y_train.push_back(y[indices[i]]);
  }
  for (size_t i = train_size; i < X.size(); ++i) {
    X_val.push_back(X[indices[i]]);
    y_val.push_back(y[indices[i]]);
  }
  regularization::EarlyStopping stoppage(patience);

  for (int epoch = 0; epoch < epochs; ++epoch) {
    for (size_t i = 0; i < X.size(); ++i) {
      std::vector<int> target(layerSizes_.back());
      for (auto j = 0; j < layerSizes_.back(); ++j) {
        target[j] = static_cast<int>(y[i][j]); // Assuming y[i] is one-hot encoded
      }
      forwardPass(X[i]);
      backwardPass(X[i], target);
      updateWeights(X[i]);
    }

    // --- Early stopping check ---
    float loss = 0.0f;
    for (size_t i = 0; i < X_val.size(); ++i) {
      std::vector<float> pred = predict(X_val[i]);
      for (size_t j = 0; j < pred.size(); ++j) {
        float diff = pred[j] - y_val[i][j];
        loss += diff * diff; // MSE loss
      }
    }
    loss /= X_val.size();

    if (stoppage.update(loss)) {
      std::cout << "\tearly stoppage, epoch " << epoch << ", loss " << loss << "\n";
      break;
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

    std::vector<float> netInputs(layers_[i].weights.rows());
#if USE_PARALLEL
    std::vector<size_t> neuronIndices(layers_[i].weights.rows());
    std::iota(neuronIndices.begin(), neuronIndices.end(), 0);
    std::for_each(std::execution::par_unseq, neuronIndices.cbegin(), neuronIndices.cend(),
      [&](size_t j)
      {
        float netInput = layers_[i].biases[j];
        
        // Vectorized dot product
        netInput += std::inner_product(layers_[i].weights[j].begin(), layers_[i].weights[j].end(), currentInput.begin(), 0.0f);
        
        switch (af_) {
        case ActivationF::Sigmoid:
          layers_[i].activations[j] = utils::sigmoid(netInput);
          break;
        case ActivationF::ReLU:
          layers_[i].activations[j] = utils::relu(netInput);
          break;
        case ActivationF::Softmax:
          netInputs[j] = netInput;
          break;
        }
      }
    );
#else
    for (size_t j = 0; j < layers_[i].weights.size(); ++j) {
      float netInput = layers_[i].biases[j];
      for (size_t k = 0; k < currentInput.size(); ++k) {
        netInput += layers_[i].weights[j][k] * currentInput[k];
      }

      switch (af_) {
      case ActivationF::Sigmoid:
        layers_[i].activations[j] = utils::sigmoid(netInput);
        break;
      case ActivationF::ReLU:
        layers_[i].activations[j] = utils::relu(netInput);
        break;
      case ActivationF::Softmax:
        netInputs[j] = netInput;
        break;
      }
    }
#endif
    if (af_ == ActivationF::Softmax) {
      if (i == layers_.size() - 1) {
        // softmax for outer layer
        layers_[i].activations = netInputs;
        utils::softmax(layers_[i].activations);
      } else {
        assert(layers_[i].activations.size() == netInputs.size());
        // sigmoid for hidden layers
        std::transform(std::execution::par_unseq, netInputs.cbegin(), netInputs.cend(), 
          layers_[i].activations.begin(), [this](float x) { return utils::sigmoid(x); });
      }
    }
    
    if (drp_ && i < layers_.size() - 1) {
      drp_->apply(layers_[i].activations, true);
    }

    currentInput = layers_[i].activations; // Pass activations
  }
}

void models::MLP::backwardPass(const std::vector<float> &input, const std::vector<int> &target) {
  assert(target.size() == layers_.back().activations.size() && "Target size must match output layer size");
  // 1. Calculate output layer deltas (error * derivative)
  for (size_t j = 0; j < layers_.back().activations.size(); ++j) {
    float output = layers_.back().activations[j];
    if (af_ == ActivationF::Sigmoid) {
      float error = output - target[j];
      layers_.back().deltas[j] = error * utils::deriveSigmoid(output);
    } else if (af_ == ActivationF::Softmax) {
      // For softmax + cross-entropy, derivative simplifies to: output - target
      layers_.back().deltas[j] = output - target[j];
    }
  }
  // 2. Calculate hidden layer deltas (propagate error backwards)
  for (int i = layers_.size() - 2; i >= 0; --i) {
    for (size_t j = 0; j < layers_[i].activations.size(); ++j) {
      float error = 0.0f;
      for (size_t k = 0; k < layers_[i + 1].deltas.size(); ++k) {
        error += layers_[i + 1].deltas[k] * layers_[i + 1].weights[k][j];
      }
      layers_[i].deltas[j] = error * utils::deriveSigmoid(layers_[i].activations[j]);
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
#if USE_PARALLEL
    std::vector<size_t> neuronIndices(layers_[i].weights.rows());
    std::iota(neuronIndices.begin(), neuronIndices.end(), 0);

    std::for_each(std::execution::par_unseq, neuronIndices.cbegin(), neuronIndices.cend(),
      [this, i, &layerInput](size_t j)
      {
        //layers_[i].biases[j] -= learningRate_ * layers_[i].deltas[j];
        opt_->update(layers_[i].biases[j], layers_[i].deltas[j]);
        std::vector<float> grads(layers_[i].weights[j].size());
        for (size_t k = 0; k < layers_[i].weights[j].size(); k++) {
          //layers_[i].weights[j][k] -= learningRate_ * layers_[i].deltas[j] * layerInput[k];
          grads[k] = layers_[i].deltas[j] * layerInput[k];
        }
        opt_->update(layers_[i].weights[j], grads, j);
        if (reg_) reg_->apply(layers_[i].weights[j]);
      });
#else
    for (size_t j = 0; j < layers_[i].weights.size(); ++j) {
      layers_[i].biases[j] -= learningRate_ * layers_[i].deltas[j];
      for (size_t k = 0; k < layers_[i].weights[j].size(); ++k) {
        layers_[i].weights[j][k] -= learningRate_ * layers_[i].deltas[j] * layerInput[k];
      }
    }
#endif
  }
}
