#include "models/svm.hpp"
#include <cassert>

models::SupportVectorMachine::SupportVectorMachine(float lr, int epochs, float C)
{
  learningRate_ = lr;
  epochs_ = epochs;
  C_ = C;
}

void models::SupportVectorMachine::fit(const std::vector<std::vector<float>> &train_x, const std::vector<int> &train_y)
{
  // Decision boundary: w·x + b = 0
  // For all samples: yᵢ(w·xᵢ + b) ≥ 1
  // Loss = 0.5 * ||w||² + C * sum(max(0, 1 - yᵢ(w·xᵢ + b)))
  // Convex QP problem : minimize `(1 / 2)

  // This is a simplified version of the SVM training process.
  // In practice, you would use a library like libsvm or scikit-learn for efficient SVM training.
  // Here we will just initialize the model parameters.
  // Initialize weights and bias

  int n_features = train_x[0].size();
  std::vector<float> weights(n_features, 0.0f);
  float bias = 0.0f;
  int n_samples = train_x.size();
  std::vector<float> gradients(n_features, 0.0f);
  std::vector<float> errors(n_samples, 0.0f);
  std::vector<int> y = train_y;
  for (int epoch = 0; epoch < epochs_; ++epoch) {
    for (int i = 0; i < n_samples; ++i) {
      float dot_product = 0.0f;
      for (int j = 0; j < n_features; ++j) {
        dot_product += weights[j] * train_x[i][j];
      }
      dot_product += bias;
      // Calculate the error
      errors[i] = y[i] * dot_product;
      // Update weights and bias
      if (errors[i] < 1.0f) {
        for (int j = 0; j < n_features; ++j) {
          gradients[j] = weights[j] - C_ * y[i] * train_x[i][j];
        }
        bias -= learningRate_ * (-C_ * y[i]);
      } else {
        for (int j = 0; j < n_features; ++j) {
          gradients[j] = weights[j];
        }
      }
      // Update weights
      for (int j = 0; j < n_features; ++j) {
        weights[j] -= learningRate_ * gradients[j];
      }
    }
  }
  weights_ = weights;
  bias_ = bias;
  // Note: This is a very basic implementation and does not include many optimizations or features of a full SVM.
}

int models::SupportVectorMachine::predict(const std::vector<float> &x) const
{
  assert(x.size() == weights_.size());
  // Calculate the dot product of weights and input vector
  float dot_product = 0.0f;
  for (size_t i = 0; i < weights_.size(); ++i) {
    dot_product += weights_[i] * x[i];
  }
  dot_product += bias_;
  return (dot_product >= 0.0f) ? 1 : -1;
}
