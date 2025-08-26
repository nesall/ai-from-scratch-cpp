#include "models/logistic_regression.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

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

void test_simple() {
  // Simple linearly separable binary classification example

  std::vector<float> x_train = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 5.0f };
  std::vector<int> y_train = { 0, 0, 0, 0, 1, 1, 1, 1 };
  Matrix<float> X(x_train.size(), 1);
  for (size_t i = 0; i < x_train.size(); i ++) {
    X(i, 0) = x_train[i];
  }

  models::LogisticRegression model(0.1f, 1000);
  model.fit(X, y_train);

  std::vector<int> predictions = model.predict(X);
  std::vector<int> expected = y_train; // Should match well if model learned correctly

  if (compare_outputs(predictions, expected)) {
    std::cout << "[LogisticRegression] 1 feature: PASSED\n";
  } else {
    std::cout << "[LogisticRegression] 1 feature: FAILED\n";
  }
}

void test_multifeature() {
  const int num_samples = 1000;
  const int num_features = 3;

  // True decision boundary: 2*x1 + 1.5*x2 - 0.8*x3 + 0.5 > 0 -> class 1
  const std::vector<float> true_weights = { 2.0f, 1.5f, -0.8f };
  const float true_bias = 0.5f;

  Matrix<float> X(num_samples, num_features);
  std::vector<int> y(num_samples);

  // Generate synthetic data
  std::random_device rd;
  std::mt19937 gen(42); // Fixed seed for reproducibility
  std::uniform_real_distribution<float> feature_dist(-2.0f, 2.0f);
  std::normal_distribution<float> noise_dist(0.0f, 0.1f);

  for (int i = 0; i < num_samples; ++i) {
    float decision_value = true_bias;

    for (int j = 0; j < num_features; ++j) {
      float feature_val = feature_dist(gen);
      X(i, j) = feature_val;
      decision_value += true_weights[j] * feature_val;
    }

    // Add small amount of noise and apply sigmoid-like probability
    decision_value += noise_dist(gen);
    float prob = 1.0f / (1.0f + std::exp(-decision_value));

    // Assign class based on probability (adds some randomness)
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    y[i] = (uniform(gen) < prob) ? 1 : 0;
  }

  // Train the model
  models::LogisticRegression model(0.1f, 2000);
  model.fit(X, y);

  // Create test cases
  Matrix<float> test_X(6, num_features);
  std::vector<int> expected_y;

  // Test case 1: Clearly class 1 - high positive decision value
  test_X(0, 0) = 1.5f; test_X(0, 1) = 1.0f; test_X(0, 2) = 0.0f;
  // Decision: 2*1.5 + 1.5*1.0 - 0.8*0.0 + 0.5 = 5.0 > 0
  expected_y.push_back(1);

  // Test case 2: Clearly class 0 - high negative decision value
  test_X(1, 0) = -1.5f; test_X(1, 1) = -1.0f; test_X(1, 2) = 2.0f;
  // Decision: 2*(-1.5) + 1.5*(-1.0) - 0.8*2.0 + 0.5 = -5.6 < 0
  expected_y.push_back(0);

  // Test case 3: Moderately class 1
  test_X(2, 0) = 1.0f; test_X(2, 1) = 0.5f; test_X(2, 2) = 0.5f;
  // Decision: 2*1.0 + 1.5*0.5 - 0.8*0.5 + 0.5 = 2.85 > 0
  expected_y.push_back(1);

  // Test case 4: Close to boundary, slightly class 0
  test_X(3, 0) = 0.0f; test_X(3, 1) = 0.0f; test_X(3, 2) = 1.0f;
  // Decision: 2*0.0 + 1.5*0.0 - 0.8*1.0 + 0.5 = -0.3 < 0
  expected_y.push_back(0);

  // Test case 5: Close to boundary, slightly class 1
  test_X(4, 0) = 0.5f; test_X(4, 1) = 0.0f; test_X(4, 2) = 0.0f;
  // Decision: 2*0.5 + 1.5*0.0 - 0.8*0.0 + 0.5 = 1.5 > 0
  expected_y.push_back(1);

  // Test case 6: Negative weights test
  test_X(5, 0) = 0.0f; test_X(5, 1) = 0.0f; test_X(5, 2) = 2.0f;
  // Decision: 2*0.0 + 1.5*0.0 - 0.8*2.0 + 0.5 = -1.1 < 0
  expected_y.push_back(0);

  auto predicted_y = model.predict(test_X);

  //std::cout << "\nTest Results:\n";
  //std::cout << "True decision boundary: " << true_weights[0] << "*x1 + "
    //<< true_weights[1] << "*x2 + " << true_weights[2] << "*x3 + "
    //<< true_bias << " > 0\n\n";

  int passed = 0;
  for (size_t i = 0; i < expected_y.size(); ++i) {
    int pred = predicted_y[i];
    int expect = expected_y[i];
    bool pass = (pred == expect);

    // Calculate true decision value for reference
    float decision = true_bias;
    for (int j = 0; j < num_features; ++j) {
      decision += true_weights[j] * test_X(i, j);
    }

    //std::cout << "Test " << (i + 1) << ": ";
    //std::cout << "x=[" << test_X(i, 0) << ", " << test_X(i, 1) << ", " << test_X(i, 2) << "] ";
    //std::cout << "decision_val=" << std::fixed << std::setprecision(2) << decision << " ";
    //std::cout << "predicted=" << pred << ", expected=" << expect;
    //std::cout << " [" << (pass ? "PASSED" : "FAILED") << "]\n";

    if (pass) passed++;

    if (pass) {
      std::cout << "[LogisticRegression] 3 features: PASSED\n";
    } else {
      std::cout << "[LogisticRegression] 3 features: FAILED\n";
    }

  }

  //std::cout << "\nOverall: " << passed << "/" << expected_y.size() << " tests passed\n";

  // Additional accuracy test on training data subset
  auto train_predictions = model.predict(X);
  int correct = 0;
  for (size_t i = 0; i < std::min(100, (int)y.size()); ++i) {
    if (train_predictions[i] == y[i]) correct++;
  }
  //std::cout << "Training accuracy (first 100 samples): " << (100.0 * correct / 100.0) << "%\n";
}

void test_logistic_regression() {
  test_simple();
  test_multifeature();
}