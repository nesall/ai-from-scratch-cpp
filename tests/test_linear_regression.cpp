#include <iostream>
#include <vector>
#include "models/linear_regression.hpp"
#include "datasynth.hpp"

void test_single_feature() {
  const int num_samples = 100;

  Matrix<float> X(num_samples, 1);
  std::vector<float> y(num_samples);
  {
    std::vector<utils::Pointf> pts;
    // True relationship: y = 2x + 1
    generate_synthetic_data<float>(pts, num_samples, 2.0f, 1.0f, 0.f, 10.f, 0.5f);
    for (size_t i = 0; i < pts.size(); i ++) {
      X(i, 0) = pts[i].x;
      y[i] = pts[i].y;
    }
  }

  models::LinearRegression model(0.001f, 10000);
  model.fit(X, y);

  // Test inputs and expected outputs
  std::vector<float> test_x = { 1.5f, 3.0f, 5.0f };
  std::vector<float> expected_y = { 2.0f * 1.5f + 1.0f, 2.0f * 3.0f + 1.0f, 2.0f * 5.0f + 1.0f }; // [4.0, 7.0, 11.0]
  Matrix<float> T(3, 1);
  for (size_t i = 0; i < test_x.size(); i ++) {
    T(i, 0) = test_x[i];
  }
  
  auto predicted_y = model.predict(T);

  for (size_t i = 0; i < test_x.size(); ++i) {
    float pred = predicted_y[i];
    float expect = expected_y[i];
    bool pass = nearly_equal(pred, expect, 0.10f);

    //std::cout << "x = " << test_x[i] << ", predicted y = " << pred << ", expected y = " << expect << "\n";
    std::cout << "[LinearRegression] 1 feature: " << (pass ? "PASSED" : "FAILED") << "\n";
  }
}

void test_multiple_features() {
  const int num_samples = 200;
  const int num_features = 3;

  // True relationship: y = 1.5*x1 + 2.0*x2 - 0.5*x3 + 3.0
  const std::vector<float> true_weights = { 1.5f, 2.0f, -0.5f };
  const float true_bias = 3.0f;

  Matrix<float> X(num_samples, num_features);
  std::vector<float> y(num_samples);

  // Generate synthetic data with some noise
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> feature_dist(-5.0f, 5.0f);
  std::normal_distribution<float> noise_dist(0.0f, 0.1f); // small noise

  for (int i = 0; i < num_samples; ++i) {
    float target = true_bias;

    for (int j = 0; j < num_features; ++j) {
      float feature_val = feature_dist(gen);
      X(i, j) = feature_val;
      target += true_weights[j] * feature_val;
    }

    // Add small amount of noise
    y[i] = target + noise_dist(gen);
  }

  // Train the model
  models::LinearRegression model(0.001f, 5000);
  model.fit(X, y);

  // Test with known inputs
  Matrix<float> test_X(4, num_features);
  std::vector<float> expected_y;

  // Test case 1: x1=1, x2=2, x3=0 -> y = 1.5*1 + 2.0*2 + (-0.5)*0 + 3 = 8.5
  test_X(0, 0) = 1.0f; test_X(0, 1) = 2.0f; test_X(0, 2) = 0.0f;
  expected_y.push_back(8.5f);

  // Test case 2: x1=0, x2=1, x3=2 -> y = 1.5*0 + 2.0*1 + (-0.5)*2 + 3 = 4.0
  test_X(1, 0) = 0.0f; test_X(1, 1) = 1.0f; test_X(1, 2) = 2.0f;
  expected_y.push_back(4.0f);

  // Test case 3: x1=2, x2=0, x3=4 -> y = 1.5*2 + 2.0*0 + (-0.5)*4 + 3 = 4.0
  test_X(2, 0) = 2.0f; test_X(2, 1) = 0.0f; test_X(2, 2) = 4.0f;
  expected_y.push_back(4.0f);

  // Test case 4: x1=-1, x2=3, x3=2 -> y = 1.5*(-1) + 2.0*3 + (-0.5)*2 + 3 = 6.5
  test_X(3, 0) = -1.0f; test_X(3, 1) = 3.0f; test_X(3, 2) = 2.0f;
  expected_y.push_back(6.5f);

  auto predicted_y = model.predict(test_X);

  //std::cout << "\nTest Results:\n";
  //std::cout << "True weights: [" << true_weights[0] << ", " << true_weights[1] << ", " << true_weights[2] << "]\n";
  //std::cout << "True bias: " << true_bias << "\n\n";

  int passed = 0;
  for (size_t i = 0; i < expected_y.size(); ++i) {
    float pred = predicted_y[i];
    float expect = expected_y[i];
    bool pass = nearly_equal(pred, expect, 0.5f); // tolerance for noise

    //std::cout << "Test " << (i + 1) << ": ";
    //std::cout << "x=[" << test_X(i, 0) << ", " << test_X(i, 1) << ", " << test_X(i, 2) << "] ";
    //std::cout << "predicted=" << pred << ", expected=" << expect;
    //std::cout << " [" << (pass ? "PASSED" : "FAILED") << "]\n";

    std::cout << "[LinearRegression] 3 features: " << (pass ? "PASSED" : "FAILED") << "\n";

    if (pass) passed++;
  }

  //std::cout << "\nOverall: " << passed << "/" << expected_y.size() << " tests passed\n";
}

void test_linear_regression() {
  test_single_feature();
  test_multiple_features();
}