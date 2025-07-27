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

void test_mlp() {
  test_xor();
}
