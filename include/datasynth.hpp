#ifndef TRAINING_DATA_1D_FOUR_CLASS_H
#define TRAINING_DATA_1D_FOUR_CLASS_H

#include "common.hpp"
#include <vector>
#include <random>


/**
 * Generates synthetic linear regression data: y = w*x + b + noise
 * @param pts Output vector for generated points
 * @param num_points Number of points to generate
 * @param true_w True slope coefficient
 * @param true_b True intercept
 * @param rangeMin Minimum x value
 * @param rangeMax Maximum x value
 * @param noise_std Standard deviation of Gaussian noise
 * @param seed Random seed for reproducible results
 */
template<typename T>
void generate_synthetic_data(std::vector<commons::Point<T>> &pts,
  int num_points,
  T true_w,
  T true_b,
  T rangeMin = static_cast<T>(0),
  T rangeMax = static_cast<T>(10),
  T noise_std = static_cast<T>(1),
  unsigned int seed = 42) {

  // Parameter validation
  if (num_points <= 0) return;
  if (rangeMin >= rangeMax) {
    std::swap(rangeMin, rangeMax);
  }
  if (noise_std < static_cast<T>(0)) {
    noise_std = static_cast<T>(0);
  }

  // Pre-allocate memory
  pts.reserve(pts.size() + num_points);

  // Setup random number generators
  std::default_random_engine eng(seed);
  std::uniform_real_distribution<T> dist_x(rangeMin, rangeMax);
  std::normal_distribution<T> dist_noise(static_cast<T>(0), noise_std);

  // Generate points
  for (int i = 0; i < num_points; ++i) {
    T xi = dist_x(eng);
    T yi = true_w * xi + true_b + dist_noise(eng);
    pts.emplace_back(xi, yi);
  }
}

template<typename T>
std::vector<commons::Point<T>> generate_synthetic_data(int num_points,
  T true_w,
  T true_b,
  T rangeMin = static_cast<T>(0),
  T rangeMax = static_cast<T>(10),
  T noise_std = static_cast<T>(1),
  unsigned int seed = 42) {

  std::vector<commons::Point<T>> pts;
  generate_synthetic_data(pts, num_points, true_w, true_b, rangeMin, rangeMax, noise_std, seed);
  return pts;
}


// Extended training data (1D, two clearly separable classes)
const std::vector<float> training_features_1D_2C = {
    0.8f, 1.0f, 1.2f, 1.5f, 1.7f, 2.0f, 2.2f, 2.5f, 3.0f, // Class 0: around 0.8–3.0
    5.8f, 6.0f, 6.2f, 6.5f, 6.8f, 7.0f, 7.2f, 7.5f, 8.0f  // Class 1: around 5.8–8.0
};
const std::vector<int> training_labels_1D_2C = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, // Corresponding labels for Class 0
    1, 1, 1, 1, 1, 1, 1, 1, 1  // Corresponding labels for Class 1
};

const std::vector<float> features_test_1D_2C = {
  0.8f, 1.2f, 1.8f,           // Points in/near Class 0 range
  3.0f, 4.0f, 5.0f,           // Points in gap between Class 0 and Class 1
  6.2f, 6.8f, 7.2f            // Points in/near Class 1 range
};
const std::vector<int> expected_1D_2C = {
  0, 0, 0,
  0, 0, 1,
  1, 1, 1
};



// Training data for a 1D classification problem with four separable classes.
// Intended for use with decision tree or random forest models.
// - x_train: Feature values (float) in 1D space, grouped into four distinct ranges.
// - y_train: Corresponding class labels (int) from 0 to 3.
// Class ranges:
//   - Class 0: Features around 0.8–3.0
//   - Class 1: Features around 5.8–8.0
//   - Class 2: Features around 10.0–12.0
//   - Class 3: Features around 14.0–16.0
const std::vector<float> training_features_1D_4C = {
    0.8f, 1.0f, 1.2f, 1.5f, 1.7f, 2.0f, 2.2f, 2.5f, 3.0f, // Class 0
    5.8f, 6.0f, 6.2f, 6.5f, 6.8f, 7.0f, 7.2f, 7.5f, 8.0f, // Class 1
    10.0f, 10.2f, 10.5f, 10.8f, 11.0f, 11.2f, 11.5f, 11.8f, 12.0f, // Class 2
    14.0f, 14.2f, 14.5f, 14.8f, 15.0f, 15.2f, 15.5f, 15.8f, 16.0f  // Class 3
};

// Corresponding labels for the training features.
// Each label (0, 1, 2, or 3) matches the feature ranges defined above.
const std::vector<int> training_labels_1D_4C = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, // Class 0 labels
    1, 1, 1, 1, 1, 1, 1, 1, 1, // Class 1 labels
    2, 2, 2, 2, 2, 2, 2, 2, 2, // Class 2 labels
    3, 3, 3, 3, 3, 3, 3, 3, 3  // Class 3 labels
};

const std::vector<float> features_test_1D_4C = {
  0.5f, 1.3f, 2.8f,           // Points in/near Class 0 range
  4.0f, 4.5f,                 // Points in gap between Class 0 and Class 1
  6.0f, 7.8f,                 // Points in/near Class 1 range
  9.0f, 9.5f,                 // Points in gap between Class 1 and Class 2
  10.5f, 11.7f,               // Points in/near Class 2 range
  13.0f, 13.5f,               // Points in gap between Class 2 and Class 3
  14.3f, 15.7f, 16.5f         // Points in/near Class 3 range
};

const std::vector<int> expected_1D_4C = {
0, 0, 0,
0, 1,
1, 1,
1, 2,
2, 2,
2, 3,
3, 3, 3
};

#endif // TRAINING_DATA_1D_FOUR_CLASS_H