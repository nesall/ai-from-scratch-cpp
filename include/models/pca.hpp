#pragma once
#include <vector>
#include "matrix.hpp"


namespace models {

  class PCA {
  public:
    PCA(int nComponents = 0);
    void fit(const Matrix<float> &X);
    Matrix<float> transform(const Matrix<float> &X, int nComponents = 0) const;
    Matrix<float> inverseTransform(const Matrix<float> &Y) const;
    std::vector<float> explainedVariance() const;
  private:
    int nComponents_;
    Matrix<float> components_; // eigenvectors (cols or rows per your convention)
    std::vector<float> explainedVariance_;
    std::vector<float> mean_;
  };

}