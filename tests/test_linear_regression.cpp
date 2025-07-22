#include <iostream>
#include <vector>
#include <cmath>
#include "models/linear_regression.hpp"
#include "datasynth.hpp"

constexpr float EPSILON = 0.2f; // Acceptable error margin for predictions

bool approximately_equal(float a, float b, float eps = EPSILON) {
  return std::fabs(a - b) <= eps;
}

void test_linear_regression() {
  std::vector<commons::Pointf> pts;
  const int num_samples = 100;

  // True relationship: y = 2x + 1
  generate_synthetic_data<float>(pts, num_samples, 2.0f, 1.0f, 0.f, 10.f, 0.5f);

  models::LinearRegression model(0.01f, 10000);
  model.fit(pts);

  // Test inputs and expected outputs
  std::vector<float> test_x = { 1.5f, 3.0f, 5.0f };
  std::vector<float> expected_y = { 2.0f * 1.5f + 1.0f, 2.0f * 3.0f + 1.0f, 2.0f * 5.0f + 1.0f }; // [4.0, 7.0, 11.0]
  auto predicted_y = model.predict(test_x);

  std::cout << "\nTest Results:\n";
  for (size_t i = 0; i < test_x.size(); ++i) {
    float pred = predicted_y[i];
    float expect = expected_y[i];
    bool pass = approximately_equal(pred, expect);

    //std::cout << "x = " << test_x[i] << ", predicted y = " << pred << ", expected y = " << expect << "\n";
    std::cout << "[LinearRegression] " << (pass ? "PASSED" : "FAILED") << '\n';
  }
}


//#include "../src/models/linear_regression.cpp"
