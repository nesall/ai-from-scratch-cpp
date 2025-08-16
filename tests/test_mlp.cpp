#include <iostream>
#include "models/mlp.hpp"

bool test_xor() {
  bool passed = true;
  // XOR inputs and targets
  std::vector<std::vector<float>> X = {
      {0, 0}, {0, 1}, {1, 0}, {1, 1}
  };
  std::vector<std::vector<float>> y = {
      {0}, {1}, {1}, {0}
  };

  // MLP with 2 inputs, one hidden layer with 4 neurons, 1 output
  models::MLP mlp({ 2, 4, 1 }, 0.5f);
  mlp.fit(X, y, 10000); // learning_rate, epochs

  for (size_t i = 0; i < X.size(); ++i) {
    auto prediction = mlp.predict(X[i])[0];
    int pred_class = prediction >= 0.5f ? 1 : 0;
    if (pred_class != static_cast<int>(y[i][0])) {
      passed = false;
    }
    std::cout << "[MLP] Input: [" << X[i][0] << ", " << X[i][1] << "] -> Predicted: "
      << pred_class << " (Raw: " << prediction << "), Target: " << y[i][0] << '\n';
  }

  std::cout << "[MLP] XOR gate: " << (passed ? "PASSED" : "FAILED") << "\n";
  return passed;
}

void test_simple2d() {
  // Create simple 2D classification problem
  std::vector<std::vector<float>> X = {
      {0.1, 0.1}, {0.1, 0.9}, {0.9, 0.1}, {0.9, 0.9},
      {0.2, 0.2}, {0.2, 0.8}, {0.8, 0.2}, {0.8, 0.8},
      {0.3, 0.1}, {0.1, 0.7}, {0.7, 0.3}, {0.9, 0.7}
  };

  std::vector<std::vector<float>> y = {
      {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0},  // Classes based on position
      {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0},
      {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}
  };

  models::MLP mlp({ 2, 10, 3 }, 0.1f, models::MLP::Initialization::Xavier);

  mlp.fit(X, y, 100, models::MLP::ActivationF::Softmax);

  int nofCorrect = 0;

  for (size_t i = 0; i < X.size(); ++i) {
    auto output = mlp.predict(X[i]);
    int predicted = 0;
    for (size_t j = 1; j < output.size(); ++j) {
      if (output[j] > output[predicted]) predicted = j;
    }

    int actual = 0;
    for (size_t j = 0; j < y[i].size(); ++j) {
      if (y[i][j] == 1.0f) actual = j;
    }

    if (predicted == actual) {
      nofCorrect++;
    }

    //std::cout << "Sample " << i << ": Pred=" << predicted << ", Actual=" << actual << std::endl;
  }
  const auto percent = (100 * nofCorrect / X.size());
  std::cout << "[MLP] Simple 2D classification: " << (75 < percent ? "PASSED" : "FAILED") << " (" << percent << "% correct)\n";
}

void test_mlp() {
  test_xor();
  test_simple2d();
}
