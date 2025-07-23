#include <iostream>
#include <vector>
#include "models/random_forest.hpp"
#include "datasynth.hpp"

void test_random_forest() {
  models::RandomForest forest(5);
  forest.fit(training_features_1D_4C, training_labels_1D_4C);

  std::vector<int> predictions = forest.predict(features_test_1D_4C);

  for (size_t i = 0; i < predictions.size(); ++i) {
    std::cout << "[RandomForest] ";
    if (predictions[i] == expected_1D_4C[i]) {
      std::cout << "PASSED\n";
    } else {
      std::cout << "FAILED\n";
    }
  }
}
