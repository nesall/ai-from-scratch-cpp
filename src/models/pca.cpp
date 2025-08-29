#include "models/pca.hpp"
#include <cassert>


models::PCA::PCA(int nComponents)
  : nComponents_(nComponents)
{
  assert(0 <= nComponents_);
}

void models::PCA::fit(const Matrix<float> &X)
{
  const int nSamples = X.rows();
  const int nFeatures = X.cols();
  assert(0 < nSamples);
  assert(0 < nFeatures);
  if (nComponents_ == 0) {
    nComponents_ = nFeatures;
  }
  assert(nComponents_ <= nFeatures);
  // 1. Center the data
  mean_.resize(nFeatures, 0.0f);
  for (int j = 0; j < nFeatures; ++j) {
    float sum = 0.0f;
    for (int i = 0; i < nSamples; ++i) {
      sum += X.at(i, j);
    }
    mean_[j] = sum / nSamples;
  }
  Matrix<float> X_centered(nSamples, nFeatures);
  for (int i = 0; i < nSamples; ++i) {
    for (int j = 0; j < nFeatures; ++j) {
      X_centered.at(i, j) = X.at(i, j) - mean_[j];
    }
  }
  // 2. Compute covariance matrix
  Matrix<float> cov = MatrixOps::covariance_matrix(X_centered);

  // 3. Eigen decomposition of covariance matrix
  auto [eigenvalues, eigenvectors] = MatrixOps::eigen(cov);

  std::vector<std::pair<float, int>> eigenPairs;
  for (int i = 0; i < eigenvalues.size(); ++i) {
    eigenPairs.emplace_back(eigenvalues[i], i);
  }
  std::sort(eigenPairs.begin(), eigenPairs.end(), std::greater<>());

  // 4. Select top nComponents eigenvectors
  components_.resize(nComponents_, nFeatures);
  explainedVariance_.resize(nComponents_);
  for (int i = 0; i < nComponents_; ++i) {
    int idx = eigenPairs[i].second;
    explainedVariance_[i] = eigenPairs[i].first;
    for (int j = 0; j < nFeatures; ++j) {
      components_.at(i, j) = eigenvectors.at(j, idx);
    }
  }
}

Matrix<float> models::PCA::transform(const Matrix<float> &X, int nComponents) const
{
  if (nComponents == -1) nComponents = nComponents_;
  assert(nComponents <= nComponents_);

  const int nSamples = X.rows();
  Matrix<float> X_centered(nSamples, X.cols());

  // Center the data
  for (int i = 0; i < nSamples; ++i) {
    for (int j = 0; j < X.cols(); ++j) {
      X_centered.at(i, j) = X.at(i, j) - mean_[j];
    }
  }

  // Project onto principal components
  auto sub = components_.submatrix(0, nComponents, 0, X.cols());
  return X_centered * sub.transpose();
}

Matrix<float> models::PCA::inverseTransform(const Matrix<float> &Y) const
{
  // Project back to original space and add mean
  Matrix<float> X_centered = Y * components_;
  Matrix<float> result(X_centered.rows(), X_centered.cols());

  for (int i = 0; i < result.rows(); ++i) {
    for (int j = 0; j < result.cols(); ++j) {
      result.at(i, j) = X_centered.at(i, j) + mean_[j];
    }
  }
  return result;
}

std::vector<float> models::PCA::explainedVariance() const
{
  return explainedVariance_;
}
