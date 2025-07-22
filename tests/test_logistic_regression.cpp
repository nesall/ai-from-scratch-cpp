#include "models/logistic_regression.hpp"
#include <iostream>
#include <vector>
#include <cmath>

// Helper to check predicted labels
bool compare_outputs(const std::vector<int> &predicted, const std::vector<int> &expected) {
  if (predicted.size() != expected.size()) return false;
  for (size_t i = 0; i < predicted.size(); ++i) {
    if (predicted[i] != expected[i]) {
      std::cerr << "Mismatch at index " << i << ": expected " << expected[i]
        << ", got " << predicted[i] << '\n';
      return false;
    }
  }
  return true;
}

int test_logistic_regression() {
  // Simple linearly separable binary classification example
  std::vector<float> x_train = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 5.0f };
  std::vector<int> y_train = { 0, 0, 0, 0, 1, 1, 1, 1 };

  LogisticRegression model(0.1f, 1000);
  model.fit(x_train, y_train);

  std::vector<int> predictions = model.predict(x_train);
  std::vector<int> expected = y_train; // Should match well if model learned correctly

  if (compare_outputs(predictions, expected)) {
    std::cout << "[LogisticRegression] PASSED\n";
  } else {
    std::cout << "[LogisticRegression] FAILED\n";
  }

  return 0;
}
