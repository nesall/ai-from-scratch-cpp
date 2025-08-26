#include <iostream>
#include <vector>
#include <cmath>
#include "models/svm.hpp"

void test_svm() {
  // Linearly separable dataset (simple binary classification)
  std::vector<std::vector<float>> X = {
      {2.0f, 3.0f},   // Class +1
      {1.0f, 1.0f},
      {2.0f, 1.0f},
      {-1.0f, -1.0f}, // Class -1
      {-2.0f, -1.0f},
      {-3.0f, -2.0f}
  };
  std::vector<int> y = { 1, 1, 1, -1, -1, -1 };

  models::SupportVectorMachine svm(/*learning_rate=*/0.01f, /*epochs=*/1000, /*C=*/1.0f);
  svm.fit(X, y);

  std::vector<int> predictions;
  for (const auto &xi : X) {
    predictions.push_back(svm.predict(xi));
  }

  bool passed = true;
  for (size_t i = 0; i < y.size(); ++i) {
    if (predictions[i] != y[i]) {
      std::cout << "FAILED: Expected " << y[i] << " but got " << predictions[i]
        << " at index " << i << std::endl;
      passed = false;
    }
  }

  if (passed) {
    std::cout << "[SVM] PASSED\n";
  } else {
    std::cout << "[SVM] FAILED\n";
  }
}
