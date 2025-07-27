#include <iostream>
#include "models/perceptron.hpp"

void test_gate(const std::string &gate_name, const std::vector<std::vector<float>> &inputs,
  const std::vector<int> &targets) {
  int expected_accuracy = targets.size();
  models::Perceptron p(2, 0.1f); // 2-input perceptron
  p.fit(inputs, targets, 20);

  int correct = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    int prediction = p.predict(inputs[i]);
    if (prediction == targets[i]) {
      ++correct;
    }
  }
  std::cout << "[Perceptron] ";
  std::cout << gate_name << ": " << (correct >= expected_accuracy ? "PASSED" : "FAILED") << "\n";
}

void test_perceptron() {
  std::vector<std::vector<float>> X = {
      {0, 0},
      {0, 1},
      {1, 0},
      {1, 1}
  };

  std::vector<int> y_and = { 0, 0, 0, 1 };
  std::vector<int> y_or = { 0, 1, 1, 1 };

  test_gate("AND gate", X, y_and);
  test_gate("OR gate", X, y_or);
}
