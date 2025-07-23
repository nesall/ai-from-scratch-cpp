#include <iostream>
#include <vector>
#include "models/decision_tree.hpp"

void test_decision_tree() {
  // Simple binary classification dataset (1D feature)
  std::vector<float> x_train = { 1.0f, 2.0f, 3.0f, 6.0f, 7.0f, 8.0f };
  std::vector<int> y_train = { 0, 0, 0, 1, 1, 1 };

  models::DecisionTree tree;
  tree.fit(x_train, y_train);

  std::vector<float> x_test = { 2.5f, 6.5f, 4.0f };
  std::vector<int> expected = { 0, 1, 0 };  // Decision boundary should split at around 4.5–5.0

  std::vector<int> predictions = tree.predict(x_test);

  for (size_t i = 0; i < predictions.size(); ++i) {
    std::cout << "[DecisionTree] ";
    if (predictions[i] == expected[i]) {
      std::cout << "PASSED\n";
    } else {
      std::cout << "FAILED\n";
    }
  }
}
