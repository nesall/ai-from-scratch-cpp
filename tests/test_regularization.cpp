#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "models/regularization.hpp"

// ======= Simple toy MLP =======
class ToyMLP {
public:
  ToyMLP(size_t input_size, size_t hidden_size, size_t output_size, float lr)
    : input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size), lr_(lr) {
    init_weights();
  }

  std::vector<float> forward(const std::vector<float> &x, bool training = true, regularization::Dropout *dropout = nullptr) {
    // hidden layer
    z1_.resize(hidden_size_);
    a1_.resize(hidden_size_);
    for (size_t j = 0; j < hidden_size_; ++j) {
      z1_[j] = 0.0f;
      for (size_t i = 0; i < input_size_; ++i) {
        z1_[j] += x[i] * W1_[j * input_size_ + i];
      }
      z1_[j] += b1_[j];
      a1_[j] = std::tanh(z1_[j]);
    }

    if (dropout) dropout->apply(a1_, training);

    // output layer (linear)
    z2_.resize(output_size_);
    for (size_t k = 0; k < output_size_; ++k) {
      z2_[k] = 0.0f;
      for (size_t j = 0; j < hidden_size_; ++j) {
        z2_[k] += a1_[j] * W2_[k * hidden_size_ + j];
      }
      z2_[k] += b2_[k];
    }
    return z2_;
  }

  void backward(const std::vector<float> &x, const std::vector<float> &y) {
    // simple MSE loss gradient
    std::vector<float> dz2(output_size_);
    for (size_t k = 0; k < output_size_; ++k) {
      dz2[k] = (z2_[k] - y[k]); // dL/dz2
    }

    // Grad for W2, b2
    for (size_t k = 0; k < output_size_; ++k) {
      for (size_t j = 0; j < hidden_size_; ++j) {
        W2_[k * hidden_size_ + j] -= lr_ * dz2[k] * a1_[j];
      }
      b2_[k] -= lr_ * dz2[k];
    }

    // Backprop into hidden
    std::vector<float> da1(hidden_size_, 0.0f);
    for (size_t j = 0; j < hidden_size_; ++j) {
      for (size_t k = 0; k < output_size_; ++k) {
        da1[j] += dz2[k] * W2_[k * hidden_size_ + j];
      }
    }

    std::vector<float> dz1(hidden_size_);
    for (size_t j = 0; j < hidden_size_; ++j) {
      dz1[j] = da1[j] * (1.0f - a1_[j] * a1_[j]); // tanh'
    }

    // Grad for W1, b1
    for (size_t j = 0; j < hidden_size_; ++j) {
      for (size_t i = 0; i < input_size_; ++i) {
        W1_[j * input_size_ + i] -= lr_ * dz1[j] * x[i];
      }
      b1_[j] -= lr_ * dz1[j];
    }
  }

  float loss(const std::vector<float> &y_pred, const std::vector<float> &y_true) {
    float sum = 0.0f;
    for (size_t k = 0; k < y_pred.size(); ++k) {
      float diff = y_pred[k] - y_true[k];
      sum += diff * diff;
    }
    return sum / y_pred.size();
  }

  // Apply regularization directly to weights
  void apply_regularization(regularization::L1 *l1, regularization::L2 *l2) {
    if (l1) {
      l1->apply(W1_);
      l1->apply(W2_);
    }
    if (l2) {
      l2->apply(W1_);
      l2->apply(W2_);
    }
  }

private:
  void init_weights() {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    W1_.resize(hidden_size_ * input_size_);
    b1_.resize(hidden_size_, 0.0f);
    W2_.resize(output_size_ * hidden_size_);
    b2_.resize(output_size_, 0.0f);
    for (auto &w : W1_) w = dist(gen);
    for (auto &w : W2_) w = dist(gen);
  }

  size_t input_size_, hidden_size_, output_size_;
  float lr_;
  std::vector<float> W1_, b1_, W2_, b2_;
  std::vector<float> z1_, a1_, z2_;
};


void test_regularization() {

  // synthetic dataset: XOR
  std::vector<std::vector<float>> X = {
      {0,0}, {0,1}, {1,0}, {1,1}
  };
  std::vector<std::vector<float>> Y = {
      {0}, {1}, {1}, {0}
  };

  ToyMLP mlp_noreg(2, 4, 1, 0.1f);
  ToyMLP mlp_reg(2, 4, 1, 0.1f);

  regularization::Dropout dropout(0.5f);
  regularization::L2 l2(0.001f);

  regularization::EarlyStopping es(5);

  const int nofEpochs = 100;
  int epoch = 0;
  for (; epoch < nofEpochs; ++epoch) {
    float total_loss_noreg = 0.0f;
    float total_loss_reg = 0.0f;

    for (size_t i = 0; i < X.size(); ++i) {
      auto y_pred1 = mlp_noreg.forward(X[i]);
      mlp_noreg.backward(X[i], Y[i]);
      total_loss_noreg += mlp_noreg.loss(y_pred1, Y[i]);

      auto y_pred2 = mlp_reg.forward(X[i], true, &dropout);
      mlp_reg.backward(X[i], Y[i]);
      mlp_reg.apply_regularization(nullptr, &l2);
      total_loss_reg += mlp_reg.loss(y_pred2, Y[i]);
    }

    if (epoch % 10 == 0) {
      std::cout << "Epoch " << epoch << " | NoReg Loss=" << total_loss_noreg << " | Reg(L2+Dropout) Loss=" << total_loss_reg << "\n";
    }

    if (es.update(total_loss_reg)) {
      //std::cout << "Early stopping triggered at epoch " << epoch << "\n";
      break;
    }

  }
  std::cout << "[regularization::EarlyStopping] " << (epoch < nofEpochs ? "PASSED" : "FAILED") << " (epoch " << epoch << " of " << nofEpochs << ")\n";

}
