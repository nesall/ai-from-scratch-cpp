#include "models/cnn.hpp"
#include "matrix.hpp"
#include <iterator>
#include <algorithm>
#include <cassert>

using namespace models;

/* WHAT CAN BE ADDED TO CNN: 
## LEVEL 1 
+ Batch normalization
+ ReLU activations in conv layers  
+ Dropout regularization
+ Better optimizers (Adam)
+ Learning rate scheduling

## LEVEL 2
+ Residual connections (ResNet-style)
+ Multiple conv layers per block
+ Different kernel sizes (1x1, 3x3, 5x5)
+ Data augmentation pipeline
+ GPU acceleration
+ Pre-trained weights

## LEVEL 3
+ Attention mechanisms
+ Multi-scale feature fusion
+ Advanced architectures (EfficientNet, Vision Transformers)
+ Quantization for mobile deployment
+ Model compression techniques
+ Robust error handling and monitoring
*/



Conv2D::Conv2D(int out_channels, int in_channels, int kernel_size, int stride, int padding)
  : out_channels_(out_channels), in_channels_(in_channels),
  kernel_size_(kernel_size), stride_(stride), padding_(padding)
{
}

void Conv2D::setWeights(const std::vector<std::vector<std::vector<std::vector<float>>>> &weights)
{
  assert(weights.size() == out_channels_ && "Weights size must match number of output channels");
  assert(!weights.empty() && !weights[0].empty() && !weights[0][0].empty() && !weights[0][0][0].empty() && 
    "Weights must not be empty and must have valid dimensions");
  weights_ = weights;
}

std::vector<std::vector<std::vector<float>>> Conv2D::forward(const std::vector<std::vector<std::vector<float>>> &input)
{
  // Input format: [in_channels][height][width]
  // Weights format: [out_channels][in_channels][kernel_height][kernel_width]
  // Output format: [out_channels][output_height][output_width]

  assert(!input.empty() && !weights_.empty());

  const int input_channels = static_cast<int>(input.size());
  const int input_height = static_cast<int>(input[0].size());
  const int input_width = static_cast<int>(input[0][0].size());

  assert(input_channels == in_channels_);

  const int output_height = (input_height - kernel_size_ + 2 * padding_) / stride_ + 1;
  const int output_width = (input_width - kernel_size_ + 2 * padding_) / stride_ + 1;

  assert(0 < output_height && 0 < output_width);

  // Initialize output tensor
  std::vector<std::vector<std::vector<float>>> output(out_channels_);
  for (int out_ch = 0; out_ch < out_channels_; ++out_ch) {
    output[out_ch].resize(output_height);
    for (int h = 0; h < output_height; ++h) {
      output[out_ch][h].resize(output_width, 0.0f);
    }
  }

  // Perform convolution for each output channel
  for (int out_ch = 0; out_ch < out_channels_; ++out_ch) {
    // For each output channel, convolve with all input channels and sum
    for (int in_ch = 0; in_ch < in_channels_; ++in_ch) {
      Matrix<float> input_matrix(input_height, input_width);
      for (int h = 0; h < input_height; ++h) {
        for (int w = 0; w < input_width; ++w) {
          input_matrix(h, w) = input[in_ch][h][w];
        }
      }
      Matrix<float> kernel_matrix(kernel_size_, kernel_size_);
      for (int kh = 0; kh < kernel_size_; ++kh) {
        for (int kw = 0; kw < kernel_size_; ++kw) {
          kernel_matrix(kh, kw) = weights_[out_ch][in_ch][kh][kw];
        }
      }
      std::vector<float> conv_result = utils::convolve2D(input_matrix, kernel_matrix, stride_, padding_);

      // Add convolution result to output channel (accumulate across input channels)
      for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
          output[out_ch][h][w] += conv_result[h * output_width + w];
        }
      }
    }
  }

  return output;
}


//------------------------------


models::MaxPool2D::MaxPool2D(int pool_size, int stride): pool_size_(pool_size), stride_(stride)
{
}

std::vector<std::vector<std::vector<float>>> MaxPool2D::forward(const std::vector<std::vector<std::vector<float>>> &input)
{
  // Input format: [channels][height][width]
  // Output format: [channels][output_height][output_width]

  assert(!input.empty());

  const int channels = static_cast<int>(input.size());
  const int input_height = static_cast<int>(input[0].size());
  const int input_width = static_cast<int>(input[0][0].size());

  const int output_height = (input_height - pool_size_) / stride_ + 1;
  const int output_width = (input_width - pool_size_) / stride_ + 1;

  assert(0 < output_height && 0 < output_width);

  // Initialize output tensor (same number of channels)
  std::vector<std::vector<std::vector<float>>> output(channels);
  for (int c = 0; c < channels; ++c) {
    output[c].resize(output_height);
    for (int h = 0; h < output_height; ++h) {
      output[c][h].resize(output_width);
    }
  }
  // Perform max pooling for each channel independently
  for (int c = 0; c < channels; ++c) {
    for (int out_h = 0; out_h < output_height; ++out_h) {
      for (int out_w = 0; out_w < output_width; ++out_w) {
        int start_h = out_h * stride_;
        int start_w = out_w * stride_;
        int end_h = start_h + pool_size_;
        int end_w = start_w + pool_size_;

        // Find maximum value in the pooling window
        float max_val = std::numeric_limits<float>::lowest();
        for (int h = start_h; h < end_h; ++h) {
          for (int w = start_w; w < end_w; ++w) {
            if (h < input_height && w < input_width) {
              max_val = std::max(max_val, input[c][h][w]);
            }
          }
        }
        output[c][out_h][out_w] = max_val;
      }
    }
  }
  return output;
}


//------------------------------

models::CNN::CNN(int in_channels, const std::vector<size_t> &layers, float learningRate) 
  : in_channels_(in_channels), current_channels_(in_channels), mlp_(layers, learningRate)
{
}

std::vector<float> CNN::forward(const std::vector<std::vector<std::vector<float>>> &input)
{
  // Input format: [channels][height][width]
  assert(!input.empty());
  assert(static_cast<int>(input.size()) == in_channels_);
  auto current_data = input;
  assert(pool_layers_.size() <= conv_layers_.size() && "Pool layers must not exceed conv layers");
  for (size_t i = 0; i < conv_layers_.size() && i < pool_layers_.size(); ++i) {
    current_data = conv_layers_[i].forward(current_data);
    current_data = pool_layers_[i].forward(current_data);
  }
  // Handle case where there are more conv layers than pool layers
  for (size_t i = pool_layers_.size(); i < conv_layers_.size(); ++i) {
    current_data = conv_layers_[i].forward(current_data);
  }
  std::vector<float> flattened = flatten(current_data);
  return mlp_.predict(flattened);
}

int CNN::predict_class(const std::vector<std::vector<std::vector<float>>> &input)
{
  std::vector<float> output = forward(input);
  assert(!output.empty());
  return static_cast<int>(std::distance(output.cbegin(), std::max_element(output.cbegin(), output.cend())));
}

std::vector<float> CNN::flatten(const std::vector<std::vector<std::vector<float>>> &input)
{
  if (input.empty()) {
    return {};
  }
  std::vector<float> flattened;
  size_t total_size = 0;
  for (const auto &channel : input) {
    for (const auto &row : channel) {
      total_size += row.size();
    }
  }
  flattened.reserve(total_size);
  // Flatten in order: [channel][height][width]
  for (const auto &channel : input) {
    for (const auto &row : channel) {
      for (float value : row) {
        flattened.push_back(value);
      }
    }
  }
  return flattened;
}

void models::CNN::add_conv_layer(int out_channels, int kernel_size, int stride, int padding)
{
  conv_layers_.push_back(Conv2D(out_channels, current_channels_, kernel_size, stride, padding));
  // Update for the next layer (conv layers change the number of channels)
  current_channels_ = out_channels;
}

void models::CNN::add_pool_layer(int pool_size, int stride)
{
  if (stride == -1) {
    stride = pool_size;
  }
  pool_layers_.emplace_back(pool_size, stride);
}

void models::CNN::set_conv_weights(size_t layer_index, const std::vector<std::vector<std::vector<std::vector<float>>>> &weights)
{
  assert(layer_index < conv_layers_.size());
  conv_layers_[layer_index].setWeights(weights);
}

std::vector<float> CNN::extract_features(const std::vector<std::vector<std::vector<float>>> &input)
{
  assert(!input.empty());
  assert(static_cast<int>(input.size()) == in_channels_);
  // Same as forward() but stop before MLP
  auto current_data = input;
  assert(pool_layers_.size() <= conv_layers_.size() && "Pool layers must not exceed conv layers");
  for (size_t i = 0; i < conv_layers_.size() && i < pool_layers_.size(); ++i) {
    current_data = conv_layers_[i].forward(current_data);
    current_data = pool_layers_[i].forward(current_data);
  }
  for (size_t i = pool_layers_.size(); i < conv_layers_.size(); ++i) {
    current_data = conv_layers_[i].forward(current_data);
  }
  return flatten(current_data);
}

void CNN::train(const std::vector<std::vector<std::vector<std::vector<float>>>> &train_images,
  const std::vector<std::vector<float>> &train_labels, size_t epochs, MLP::ActivationF activation)
{
  assert(train_images.size() == train_labels.size());
  assert(!train_images.empty());
  std::vector<std::vector<float>> features;
  features.reserve(train_images.size());
  for (size_t i = 0; i < train_images.size(); ++i) {
    features.push_back(extract_features(train_images[i]));
  }
  mlp_.fit(features, train_labels, epochs, activation);
}