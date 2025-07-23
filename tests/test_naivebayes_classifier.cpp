#include <iostream>
#include <vector>
#include "models/naivebayes_classifier.hpp"

void test_naivebayes_classification() {
  // Training data (1D features)
  std::vector<float> x_train = { 1.0f, 2.0f, 1.5f, 6.0f, 7.0f, 6.5f };
  std::vector<int> y_train = { 0, 0, 0, 1, 1, 1 };

  // Test data
  std::vector<float> x_test = { 1.2f, 6.8f, 3.9f };
  std::vector<int> expected = { 0, 1, 0 };

  models::GaussianNaiveBayes gnb;
  gnb.fit(x_train, y_train);

  std::vector<int> predictions = gnb.predict(x_test);

  for (size_t i = 0; i < predictions.size(); ++i) {
    std::cout << "[GaussianNaiveBayes] ";
    if (predictions[i] == expected[i]) {
      std::cout << "PASSED\n";
    } else {
      std::cout << "FAILED\n";
    }
  }
}
