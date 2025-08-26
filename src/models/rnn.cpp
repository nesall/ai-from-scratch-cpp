#include "models/rnn.hpp"
#include "models/optimizers.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <iostream>
#include <execution>


models::RNN::RNN(int input_size, int hidden_size, int output_size, bool initW) :
  input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size)
{
  assert(input_size > 0 && "Input size must be greater than 0");
  assert(hidden_size > 0 && "Hidden size must be greater than 0");
  assert(output_size > 0 && "Output size must be greater than 0");

  // Initialize weight matrices (will be properly initialized later via initialize_weights)
  Wxh_.resize(hidden_size_, input_size_);   // [hidden_size x input_size]
  Whh_.resize(hidden_size_, hidden_size_);  // [hidden_size x hidden_size]
  Why_.resize(output_size_, hidden_size_);  // [output_size x hidden_size]

  // Initialize bias vectors
  bh_.resize(hidden_size_, 0.0f);   // hidden bias
  by_.resize(output_size_, 0.0f);   // output bias

  // Initialize state
  h_prev_.resize(hidden_size_, 0.0f);  // previous hidden state

  // Initialize gradients (same structure as weights)
  dWxh_.resize(hidden_size_, input_size_); // [hidden_size x input_size]
  dWhh_.resize(hidden_size_, hidden_size_); // [hidden_size x hidden_size]
  dWhy_.resize(output_size_, hidden_size_); // [output_size x hidden_size]

  dbh_.resize(hidden_size_, 0.0f);
  dby_.resize(output_size_, 0.0f);

  if (initW)
    initialize_weights(Initialization::Xavier);

}

void models::RNN::reset_state()
{
  std::fill(h_prev_.begin(), h_prev_.end(), 0.0f);
}

void models::RNN::fit(const std::vector<std::vector<std::vector<float>>> &sequences, const std::vector<std::vector<std::vector<float>>> &targets, int epochs, float learning_rate, ActivationF hidden_af, ActivationF output_af)
{
  current_hidden_af_ = hidden_af;
  current_output_af_ = output_af;

  assert(!sequences.empty() && "Sequences cannot be empty");

  for (int epoch = 0; epoch < epochs; ++epoch) {
    float total_loss = 0.0f;
    for (size_t i = 0; i < sequences.size(); ++i) {
      float loss = train_sequence(Matrix<float>(sequences[i]), Matrix<float>(targets[i]), learning_rate);
      total_loss += loss;
    }
  }
}

std::vector<std::vector<float>> models::RNN::predict(const std::vector<std::vector<float>> &sequence)
{
  reset_state();
  return forward(Matrix(sequence));
}

float models::RNN::train_sequence(const Matrix<float> &sequence, const Matrix<float> &targets, float learning_rate)
{
  if (sequence.nofElems() > 1'000'000) {
    throw std::runtime_error("Sequence length exceeds maximum allowed");
  }
  zero_gradients();
  auto predictions = forward_training(sequence);
  auto loss = compute_loss(predictions, targets);
  backward_through_time(targets);
  update_weights(learning_rate);
  return loss;
}

float models::RNN::compute_loss(const Matrix<float> &predictions, const Matrix<float> &targets)
{
  assert(!predictions.empty() && "Predictions cannot be empty");
  assert(predictions.rows() == targets.rows() && "Predictions and targets must have the same size");
  float loss = 0.0f;
  if (current_output_af_ == ActivationF::Softmax) {
    for (size_t i = 0; i < predictions.rows(); ++i) {
      for (size_t j = 0; j < predictions[i].size(); ++j) {
        loss -= targets[i][j] * std::log(std::max(predictions[i][j], 1e-10f)); // Avoid log(0)
      }
    }
    loss /= predictions.rows();
  } else {
    // MSE Loss Calculation
    for (size_t i = 0; i < predictions.rows(); ++i) {
      for (size_t j = 0; j < predictions[i].size(); ++j) {
        float diff = std::clamp(predictions[i][j] - targets[i][j], -1e5f, 1e5f);
        loss += diff * diff;
      }
    }
    loss /= predictions.rows() * predictions[0].size();
  }
  return loss;
}

void models::RNN::initialize_weights(Initialization method)
{
  WeightInitializer::initialize_matrix(Wxh_, hidden_size_, input_size_, method);
  WeightInitializer::initialize_matrix(Whh_, hidden_size_, hidden_size_, method);
  WeightInitializer::initialize_matrix(Why_, output_size_, hidden_size_, method);
}

std::vector<std::vector<float>> models::RNN::forward(const Matrix<float> &sequence)
{
  if (sequence.empty()) {
    return std::vector<std::vector<float>>(0, std::vector<float>(output_size_, 0.0f));
  }
  std::vector<std::vector<float>> res;
  res.reserve(sequence.rows());
  std::vector<float> hidden_raw(hidden_size_);
  std::vector<float> hidden(hidden_size_);
  std::vector<float> output_raw(output_size_);
  std::vector<float> output(output_size_);
  for (auto input : sequence) {
    forward_timestep(input, h_prev_, hidden_raw, hidden, output_raw, output);
    res.push_back(output);
    h_prev_ = hidden;
  }
  timesteps_.clear();
  if (res.empty()) {
    res.resize(sequence.rows(), std::vector<float>(output_size_, 0.0f));
  }
  return res;
}

Matrix<float> models::RNN::forward_training(const Matrix<float> &sequence)
{
  if (sequence.empty()) {
    return std::vector<std::vector<float>>(0, std::vector<float>(output_size_, 0.0f));
  }
  std::vector<std::vector<float>> res;
  res.resize(sequence.rows());
  std::vector<float> hidden_raw(hidden_size_);
  std::vector<float> hidden(hidden_size_);
  std::vector<float> output_raw(output_size_);
  std::vector<float> output(output_size_);
  int k = 0;
  for (auto input : sequence) {
    forward_timestep(input, h_prev_, hidden_raw, hidden, output_raw, output);
    timesteps_.push_back({ input, hidden_raw, hidden, output_raw, output, h_prev_ });
    res[k++] = std::move(output);
    h_prev_ = std::move(hidden);
  }
  assert(res.size() == sequence.rows());
  return res;
}

void models::RNN::forward_timestep(
  std::span<const float> input, const std::vector<float> &h_prev, std::vector<float> &hidden_raw, std::vector<float> &hidden, 
  std::vector<float> &output_raw, std::vector<float> &output)
{
  assert(input.size() == input_size_ && "Input size mismatch");
  assert(h_prev.size() == hidden_size_ && "Hidden state size mismatch");
  assert(hidden_raw.empty() || hidden_raw.size() == hidden_size_);
  assert(hidden.empty() || hidden.size() == hidden_size_);
  assert(output_raw.empty() || output_raw.size() == output_size_);
  assert(output.empty() || output.size() == output_size_);

  // Resize output vectors
  hidden_raw.resize(hidden_size_);
  hidden.resize(hidden_size_);
  output_raw.resize(output_size_);
  output.resize(output_size_);

  // 1. Compute hidden state: h = activation(Wxh * x + Whh * h_prev + bh)

  // Wxh * x (input to hidden)
  auto wxh_x{ matvec_mul(Wxh_, input) };

  // Whh * h_prev (hidden to hidden)
  auto whh_h{ matvec_mul(Whh_, h_prev) };

  // Wxh * x + Whh * h_prev + bh
  for (int i = 0; i < hidden_size_; ++i) {
    hidden_raw[i] = wxh_x[i] + whh_h[i] + bh_[i];
  }

  // Apply hidden activation function
  hidden = apply_activation(hidden_raw, current_hidden_af_);

  // 2. Compute output: y = activation(Why * h + by)

  // Why * h (hidden to output)
  auto why_h = matvec_mul(Why_, hidden);

  // Why * h + by
  for (int i = 0; i < output_size_; ++i) {
    output_raw[i] = why_h[i] + by_[i];
  }

  // Apply output activation function
  output = apply_activation(output_raw, current_output_af_);
  if (current_output_af_ == ActivationF::Softmax) {
    float sum = std::accumulate(output.begin(), output.end(), 0.0f);
    assert(std::abs(sum - 1.0f) < 1e-5f && "Softmax output must sum to 1");
  }
}

void models::RNN::backward_through_time(const Matrix<float> &targets)
{
  assert(targets.rows() == timesteps_.size() && "Targets must match timesteps");
  for (const auto &target : targets) {
    assert(target.size() == output_size_ && "Target size must match output size");
  }

  std::vector<float> dh_next(hidden_size_, 0.0f);

  for (int t = static_cast<int>(timesteps_.size()) - 1; t >= 0; --t) {
    const auto &[x, h_raw, h, y_raw, y, h_prev] = timesteps_[t];

    // 1. Output layer gradient computation
    std::vector<float> dy_raw(output_size_);
    if (current_output_af_ == ActivationF::Softmax) {
      // For softmax with cross-entropy loss: dy_raw = y - targets
      for (int i = 0; i < output_size_; ++i) {
        dy_raw[i] = y[i] - targets[t][i];
      }
    } else {
      // For other activations with MSE loss: chain rule
      for (int i = 0; i < output_size_; ++i) {
        float loss_grad = 2.0f * (y[i] - targets[t][i]) / static_cast<float>(timesteps_.size() * output_size_); // MSE derivative
        float activation_grad = activation_derivative({ y_raw[i] }, current_output_af_)[0];
        dy_raw[i] = loss_grad * activation_grad;
      }
    }

    // 2. Accumulate gradients for output layer weights (Why, by)
    for (int i = 0; i < output_size_; ++i) {
      dby_[i] += dy_raw[i];
      for (int j = 0; j < hidden_size_; ++j) {
        dWhy_[i][j] += dy_raw[i] * h[j];
      }
    }

    // 3. Backpropagate to hidden layer
    std::vector<float> dh_total(hidden_size_, 0.0f);

    // Gradient from output layer
    for (int i = 0; i < hidden_size_; ++i) {
      for (int j = 0; j < output_size_; ++j) {
        dh_total[i] += dy_raw[j] * Why_[j][i];
      }
    }

    // Add gradient from next timestep
    for (int i = 0; i < hidden_size_; ++i) {
      dh_total[i] += dh_next[i];
    }

    // 4. Apply hidden activation derivative
    auto dh_raw_vec = activation_derivative(h_raw, current_hidden_af_);
    std::vector<float> dh_raw(hidden_size_);
    for (int i = 0; i < hidden_size_; ++i) {
      dh_raw[i] = dh_total[i] * dh_raw_vec[i];

      // Optional gradient clipping
      const float clip_threshold = 5.0f;
      dh_raw[i] = std::clamp(dh_raw[i], -clip_threshold, clip_threshold);
    }

    // 5. Accumulate gradients for hidden layer weights and biases
    for (int i = 0; i < hidden_size_; ++i) {
      // Bias gradient
      dbh_[i] += dh_raw[i];

      // Input-to-hidden weights (Wxh)
      for (int j = 0; j < input_size_; ++j) {
        dWxh_[i][j] += dh_raw[i] * x[j];
      }

      // Hidden-to-hidden weights (Whh)
      for (int j = 0; j < hidden_size_; ++j) {
        dWhh_[i][j] += dh_raw[i] * h_prev[j];
      }
    }

    // 6. Compute gradient for next timestep (going backwards)
    dh_next.assign(hidden_size_, 0.0f);
    for (int i = 0; i < hidden_size_; ++i) {
      for (int j = 0; j < hidden_size_; ++j) {
        dh_next[i] += dh_raw[j] * Whh_[j][i];
      }
    }
  }
}

void models::RNN::update_weights(float learning_rate)
{
  assert(Wxh_.cols() == input_size_);
  assert(Wxh_.rows() == hidden_size_);
  assert(Whh_.cols() == hidden_size_);
  assert(Whh_.rows() == hidden_size_);
  assert(Why_.cols() == hidden_size_);
  assert(Why_.rows() == output_size_);
  optimizers::Adam opt(learning_rate, 0.9f, 0.9f);
  for (int i = 0; i < hidden_size_; ++i) {
    opt.update(Wxh_[i], dWxh_[i]);
    opt.update(Whh_[i], dWhh_[i]);
  }
  opt.update(bh_, dbh_);
  for (int i = 0; i < output_size_; ++i) {
    opt.update(Why_[i], dWhy_[i]);
  }
  opt.update(by_, dby_);
}

void models::RNN::zero_gradients()
{
  dWxh_.fill(0.0f);
  dWhh_.fill(0.0f);
  dWhy_.fill(0.0f);
  std::fill(dbh_.begin(), dbh_.end(), 0.0f);
  std::fill(dby_.begin(), dby_.end(), 0.0f);
  timesteps_.clear(); // Clear stored timesteps
}

std::vector<float> models::RNN::matvec_mul(const Matrix<float> &W, std::span<const float> x)
{
  return MatrixOps::matvec_mul(W, x);
}

std::vector<float> models::RNN::matvec_mul(const Matrix<float> &W, const std::vector<float> &x)
{
  return MatrixOps::matvec_mul(W, x);
}

std::vector<float> models::RNN::add(const std::vector<float> &a, const std::vector<float> &b)
{
  return MatrixOps::add(a, b);
}

std::vector<float> models::RNN::tanh_vec(const std::vector<float> &x)
{
  return apply_activation(x, ActivationF::Tanh);
}

std::vector<float> models::RNN::apply_activation(const std::vector<float> &x, ActivationF af)
{
  return ActivationFunctions::apply(x, af);
}

std::vector<float> models::RNN::activation_derivative(const std::vector<float> &x, ActivationF af)
{
  return ActivationFunctions::derivative(x, af);
}

Matrix<float> models::RNN::numerical_gradient(
    models::RNN &rnn,
    const Matrix<float> &sequence,
    const Matrix<float> &targets,
    Matrix<float> &weight_matrix,
    float epsilon)
{
  std::vector<std::vector<float>> num_grad(weight_matrix.rows(), std::vector<float>(weight_matrix[0].size()));
  float original_loss = rnn.train_sequence(sequence, targets, 0.0f); // No weight update

  for (size_t i = 0; i < weight_matrix.rows(); ++i) {
    for (size_t j = 0; j < weight_matrix[i].size(); ++j) {
      // Perturb weight positively
      float original_weight = weight_matrix[i][j];
      weight_matrix[i][j] += epsilon;
      rnn.reset_state();
      float loss_plus = rnn.train_sequence(sequence, targets, 0.0f);

      // Perturb weight negatively
      rnn.reset_state();
      weight_matrix[i][j] = original_weight - epsilon;
      float loss_minus = rnn.train_sequence(sequence, targets, 0.0f);

      // Compute numerical gradient
      num_grad[i][j] = (loss_plus - loss_minus) / (2 * epsilon);

      // Restore original weight
      weight_matrix[i][j] = original_weight;
    }
  }
  return num_grad;
}

void models::RNN::run_basic_test()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

  // Initialize RNN: input_size=3, hidden_size=4, output_size=2
  models::RNN rnn(3, 4, 2);
  rnn.initialize_weights(models::Initialization::Xavier);

  // Create a batch of 2 sequences, each of length 3
  std::vector<std::vector<std::vector<float>>> sequences = {
    // Sequence 1
    {
      {0.1f, 0.2f, 0.3f},
      {0.4f, 0.5f, 0.6f},
      {0.7f, 0.8f, 0.9f}
    },
    // Sequence 2
    {
      {0.2f, 0.3f, 0.4f},
      {0.5f, 0.6f, 0.7f},
      {0.8f, 0.9f, 0.1f}
    }
  };

  // Create corresponding targets
  std::vector<std::vector<std::vector<float>>> targets = {
    // Targets for sequence 1
    {
      {0.7f, 0.3f},
      {0.6f, 0.4f},
      {0.5f, 0.5f}
    },
    // Targets for sequence 2
    {
      {0.4f, 0.6f},
      {0.3f, 0.7f},
      {0.2f, 0.8f}
    }
  };

  bool test_passed = true;

  // Test 1: Forward pass with Tanh/Sigmoid
  rnn.reset_state();
  auto outputs = rnn.forward(sequences[0]);
  assert(outputs.size() == 3 && "Output sequence length mismatch");
  for (const auto &output : outputs) {
    assert(output.size() == 2 && "Output size mismatch");
    for (float val : output) {
      assert(!std::isnan(val) && !std::isinf(val) && "Invalid output value");
    }
  }
  std::cout << "[RNN] Forward pass: " << (test_passed ? "PASSED" : "FAILED") << std::endl;

  // Test 2: Loss computation
  float loss = rnn.compute_loss(outputs, targets[0]);
  assert(!std::isnan(loss) && !std::isinf(loss) && loss >= 0.0f && "Invalid loss value");
  std::cout << "[RNN] Initial loss for sequence 1: " <<  (0 <= loss && loss < 0.1 ? "PASSED" : "FAILED (loss " + std::to_string(loss) + ")") << std::endl;

  // Test 3: Gradient computation
  rnn.reset_state();
  rnn.forward_training(sequences[0]);
  rnn.backward_through_time(targets[0]);

  // Check gradients
  for (const auto &row : rnn.dWxh_) {
    for (float val : row) {
      assert(!std::isnan(val) && !std::isinf(val) && "Invalid dWxh gradient");
    }
  }
  for (const auto &row : rnn.dWhh_) {
    for (float val : row) {
      assert(!std::isnan(val) && !std::isinf(val) && "Invalid dWhh gradient");
    }
  }
  for (const auto &row : rnn.dWhy_) {
    for (float val : row) {
      assert(!std::isnan(val) && !std::isinf(val) && "Invalid dWhy gradient");
    }
  }
  for (float val : rnn.dbh_) {
    assert(!std::isnan(val) && !std::isinf(val) && "Invalid dbh gradient");
  }
  for (float val : rnn.dby_) {
    assert(!std::isnan(val) && !std::isinf(val) && "Invalid dby gradient");
  }
  std::cout << "[RNN] Gradients computed:" << (test_passed ? "PASSED" : "FAILED") << std::endl;

  // Test 4: Numerical gradient checking for Wxh
  auto num_grad = numerical_gradient(rnn, sequences[0], targets[0], rnn.Wxh_, 1e-4);
  float grad_error = 0.0f;
  for (size_t i = 0; i < rnn.dWxh_.rows(); ++i) {
    for (size_t j = 0; j < rnn.dWxh_[i].size(); ++j) {
      grad_error += std::abs(rnn.dWxh_[i][j] - num_grad[i][j]);
    }
  }
  grad_error /= rnn.dWxh_.rows() * rnn.dWxh_[0].size();
  assert(grad_error < 1e-3 && "Numerical gradient check failed for Wxh");
  std::cout << "[RNN] Numerical gradient check for Wxh: " << (test_passed ? "PASSED" : "FAILED") << std::endl;


  // Test 5: Batch training with fit
  float initial_loss = 0.0f;
  for (size_t i = 0; i < sequences.size(); ++i) {
    rnn.reset_state();
    initial_loss += rnn.compute_loss(rnn.forward(sequences[i]), targets[i]);
  }
  initial_loss /= sequences.size();

  rnn.fit(sequences, targets, 5, 0.01f, models::ActivationF::Tanh, models::ActivationF::Sigmoid);
  float final_loss = 0.0f;
  for (size_t i = 0; i < sequences.size(); ++i) {
    rnn.reset_state();
    final_loss += rnn.compute_loss(rnn.forward(sequences[i]), targets[i]);
  }
  final_loss /= sequences.size();
  //assert(final_loss < initial_loss && "Loss did not decrease after training");
  std::cout << "[RNN] Loss decreased from " << initial_loss << " to " << final_loss << ": " << (final_loss < initial_loss ? "PASSED" : "FAILED") << std::endl;

  // Test 6: Test with Softmax output
  rnn.reset_state();
  rnn.fit(sequences, targets, 1, 0.01f, models::ActivationF::Tanh, models::ActivationF::Softmax);
  auto softmax_outputs = rnn.forward(sequences[0]);
  for (const auto &output : softmax_outputs) {
    float sum = std::accumulate(output.begin(), output.end(), 0.0f);
    assert(std::abs(sum - 1.0f) < 1e-5f && "Softmax outputs must sum to 1");
  }
  std::cout << "[RNN] Softmax outputs: " << (test_passed ? "PASSED" : "FAILED") << std::endl;

  // Test 7: Test with ReLU hidden activation
  rnn.reset_state();
  rnn.initialize_weights(models::Initialization::He); // He initialization for ReLU
  rnn.fit(sequences, targets, 1, 0.01f, models::ActivationF::ReLU, models::ActivationF::Sigmoid);
  auto relu_outputs = rnn.forward(sequences[0]);
  for (const auto &output : relu_outputs) {
    for (float val : output) {
      assert(!std::isnan(val) && !std::isinf(val) && "Invalid ReLU output");
    }
  }
  std::cout << "[RNN] ReLU hidden activation: " << (test_passed ? "PASSED" : "FAILED") << std::endl;
}