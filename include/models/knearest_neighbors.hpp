#pragma once

#include <vector>

namespace models {

  class KNearestNeighbors {
  public:
    KNearestNeighbors(int k);
    void fit(const std::vector<float> &x_train, const std::vector<int> &y_train);
    int predict(float x_test);
    std::vector<int> predict(const std::vector<float> &x_tests);

  private:
    int k_ = 0;
    std::vector<float> x_train_;
    std::vector<int> y_train_;
  };

}