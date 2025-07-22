#include <iostream>
#include <vector>
#include "models/knearest_neighbors.hpp"

void test_knn_classification() {
  std::vector<float> x_train = { 1.0f, 2.0f, 3.0f, 6.0f, 7.0f, 8.0f };
  std::vector<int> y_train = { 0, 0, 0, 1, 1, 1 };

  models::KNearestNeighbors knn(3);
  knn.fit(x_train, y_train);

  std::vector<float> x_test = { 2.5f, 6.5f, 4.5f, 4.6f };
  std::vector<int> expected = { 0, 1, 0, 1 };  // depending on k

  std::vector<int> predictions = knn.predict(x_test);

  for (size_t i = 0; i < predictions.size(); ++i) {
    std::cout << "[KNearestNeighbors] ";
    if (predictions[i] == expected[i]) {
      std::cout << "PASSED\n";
    } else {
      std::cout << "FAILED\n";
    }
  }
}
