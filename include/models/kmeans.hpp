#pragma once
#include <vector> 
#include "matrix.hpp"


namespace models {

  class KMeans {
  public:
    KMeans(int k, int maxIters = 300, bool use_kmeanspp = true);
    void fit(const Matrix<float> &X);
    std::vector<int> predict(const Matrix<float> &X) const;
    const Matrix<float> &centroids() const;
    float inertia() const; // WCSS
  private:
    int k_;
    int maxIters_;
    bool kpp_;
    Matrix<float> centroids_;
    float inertia_;
  };

}