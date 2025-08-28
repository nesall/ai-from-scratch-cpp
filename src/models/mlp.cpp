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
  size_t epochs, ActivationF afOutput, ActivationF afHidden, float validationRatio, size_t patience)
{
  assert(X.size() == y.size() && "Input and target sizes must match");
  assert(!X.empty() && X[0].size() == layerSizes_[0] && "Input size must match first layer size");
  assert(layerSizes_.back() == y[0].size() && "Output size must match last layer size");
  afForOutputLayers_ = afOutput;
  afForHiddenLayers_ = afHidden;

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
    for (size_t i = 0; i < X_train.size(); ++i) {
      std::vector<int> target(layerSizes_.back());
      for (auto j = 0; j < layerSizes_.back(); ++j) {
        target[j] = static_cast<int>(y_train[i][j]); // Assuming y[i] is one-hot encoded
      }
      forwardPass(X_train[i]);
      backwardPass(X_train[i], target);
      updateWeights(X_train[i]);
    }

    if (debug_) {
      // DEBUG: Check for dead ReLU neurons
      if (epoch % 5 == 0) {  // Every 5 epochs
        for (size_t layer_idx = 0; layer_idx < layers_.size() - 1; ++layer_idx) {  // Skip output layer
          int dead_count = 0;
          for (float activation : layers_[layer_idx].activations) {
            if (activation == 0.0f) dead_count++;
          }
          std::cout << "Epoch " << epoch << ", Layer " << layer_idx
            << ": " << dead_count << "/" << layers_[layer_idx].activations.size()
            << " dead neurons" << std::endl;
        }

        // DEBUG: Check weight magnitudes
        float max_weight = 0.0f, min_weight = 0.0f;
        for (const auto &layer : layers_) {
          for (size_t i = 0; i < layer.weights.rows(); ++i) {
            for (size_t j = 0; j < layer.weights.cols(); ++j) {
              max_weight = std::max(max_weight, layer.weights[i][j]);
              min_weight = std::min(min_weight, layer.weights[i][j]);
            }
          }
        }
        std::cout << "Epoch " << epoch << ", Weight range: [" << min_weight << ", " << max_weight << "]" << std::endl;

        // DEBUG: Check output layer activations for a sample
        std::vector<float> sample_pred = predict(X_train[0]);
        std::cout << "Epoch " << epoch << ", Sample prediction: ";
        for (size_t i = 0; i < sample_pred.size(); ++i) {
          std::cout << sample_pred[i] << " ";
        }
        std::cout << std::endl;
      }
    } // debug

    // --- Early stopping check ---
    float loss = 0.0f;
    for (size_t i = 0; i < X_val.size(); ++i) {
      std::vector<float> pred = predict(X_val[i]);
      for (size_t j = 0; j < pred.size(); ++j) {
        if (afForOutputLayers_ == ActivationF::Softmax) { // Cross-entropy loss
          float target_val = y_val[i][j];
          if (target_val > 0) {  // Only for the correct class
            loss -= target_val * std::log(std::max(pred[j], 1e-15f));
          }
        } else { // MSE loss
          float diff = pred[j] - y_val[i][j];
          loss += diff * diff;
        }
      }
    }
    loss /= X_val.size();

    if (stoppage.update(loss)) {
      std::cout << "\tearly stoppage, epoch " << epoch << ", loss " << loss << "\n";
      break;
    }

    if (debug_) {
      if (epoch % 5 == 0 || epoch == epochs - 1) {
        std::cout << "Epoch " << epoch << ", Validation Loss: " << loss << std::endl;
      }
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
    const bool isOutputLayer = (i == layers_.size() - 1);
    const ActivationF currentAF = isOutputLayer ? afForOutputLayers_ : afForHiddenLayers_;

    std::vector<float> netInputs(layers_[i].weights.rows());
    std::vector<size_t> neuronIndices(layers_[i].weights.rows());
    std::iota(neuronIndices.begin(), neuronIndices.end(), 0);
    std::for_each(std::execution::par_unseq, neuronIndices.cbegin(), neuronIndices.cend(),
      [&](size_t j)
      {
        float netInput = layers_[i].biases[j];        
        netInput += std::inner_product(layers_[i].weights[j].begin(), layers_[i].weights[j].end(), currentInput.begin(), 0.0f);
        netInputs[j] = netInput;
      }
    );
    // Apply activation function
    if (currentAF == ActivationF::Softmax) {
      layers_[i].activations = utils::softmax(netInputs);
    } else {
      std::transform(std::execution::par_unseq, netInputs.cbegin(), netInputs.cend(),
        layers_[i].activations.begin(),
        [currentAF](float x) {
          switch (currentAF) {
          case ActivationF::Sigmoid:
            return utils::sigmoid(x);
          case ActivationF::ReLU:
            return utils::leakyRelu(x);
          default:
            return x;
          }
        });
    }

    if (drp_ && !isOutputLayer) {
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
    if (afForOutputLayers_ == ActivationF::Sigmoid) {
      float error = output - target[j];
      layers_.back().deltas[j] = error * utils::deriveSigmoid(output);
    } else if (afForOutputLayers_ == ActivationF::Softmax) {
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
      //layers_[i].deltas[j] = error * utils::deriveSigmoid(layers_[i].activations[j]);// For hidden layers:
      if (afForHiddenLayers_ == ActivationF::ReLU) {
        layers_[i].deltas[j] = error * utils::deriveReLU(layers_[i].activations[j]);
      } else if (afForHiddenLayers_ == ActivationF::Sigmoid) {
        layers_[i].deltas[j] = error * utils::deriveSigmoid(layers_[i].activations[j]);
      }
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
    std::vector<size_t> neuronIndices(layers_[i].weights.rows());
    std::iota(neuronIndices.begin(), neuronIndices.end(), 0);

    std::for_each(std::execution::par_unseq, neuronIndices.cbegin(), neuronIndices.cend(),
      [this, i, &layerInput](size_t j)
      {
        opt_->update(layers_[i].biases[j], layers_[i].deltas[j]);
        std::vector<float> grads(layers_[i].weights[j].size());
        for (size_t k = 0; k < layers_[i].weights[j].size(); k++) {
          grads[k] = layers_[i].deltas[j] * layerInput[k];
        }
        opt_->update(layers_[i].weights[j], grads, j);
        if (reg_) reg_->apply(layers_[i].weights[j]);
      });

    if (i == layers_.size() - 1) {  // Only for the last layer
      opt_->incrementStep();
    }
  }
}
