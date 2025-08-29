#include "models/kmeans.hpp"
#include <random>


models::KMeans::KMeans(int k, int maxIters, bool use_kmeanspp)
  : k_(k), maxIters_(maxIters), kpp_(use_kmeanspp), inertia_(0.0f)
{
  assert(0 < k_);
  assert(0 < maxIters_);
}

void models::KMeans::fit(const Matrix<float> &X)
{
  const int nSamples = X.rows();
  const int nFeatures = X.cols();
  assert(k_ <= nSamples);
  // Initialize centroids
  centroids_.resize(k_, nFeatures);
  std::mt19937 gen(42); // fixed seed for reproducibility
  if (kpp_) {
    // KMeans++ initialization
    std::uniform_int_distribution<int> dis(0, nSamples - 1);
    int first_idx = dis(gen);
    for (int j = 0; j < nFeatures; ++j) {
      centroids_.at(0, j) = X.at(first_idx, j);
    }
    std::vector<float> distances(nSamples, std::numeric_limits<float>::max());
    for (int c = 1; c < k_; ++c) {
      // Update distances to nearest centroid
      for (int i = 0; i < nSamples; ++i) {
        float dist = 0.0f;
        for (int j = 0; j < nFeatures; ++j) {
          float diff = X.at(i, j) - centroids_.at(c - 1, j);
          dist += diff * diff;
        }
        if (dist < distances[i]) {
          distances[i] = dist;
        }
      }
      // Choose next centroid weighted by squared distance
      std::discrete_distribution<int> weighted_dis(distances.begin(), distances.end());
      int next_idx = weighted_dis(gen);
      for (int j = 0; j < nFeatures; ++j) {
        centroids_.at(c, j) = X.at(next_idx, j);
      }
    }
  } else {
    // Random initialization
    std::uniform_int_distribution<int> dis(0, nSamples - 1);
    std::vector<int> chosen;
    while (chosen.size() < static_cast<size_t>(k_)) {
      int idx = dis(gen);
      if (std::find(chosen.begin(), chosen.end(), idx) == chosen.end()) {
        chosen.push_back(idx);
        for (int j = 0; j < nFeatures; ++j) {
          centroids_.at(chosen.size() - 1, j) = X.at(idx, j);
        }
      }
    }
  }
  std::vector<int> labels(nSamples, -1);
  for (int iter = 0; iter < maxIters_; ++iter) {
    bool changed = false;
    // Assignment step
    for (int i = 0; i < nSamples; ++i) {
      float best_dist = std::numeric_limits<float>::max();
      int best_cluster = -1;
      for (int c = 0; c < k_; ++c) {
        float dist = 0.0f;
        for (int j = 0; j < nFeatures; ++j) {
          float diff = X.at(i, j) - centroids_.at(c, j);
          dist += diff * diff;
        }
        if (dist < best_dist) {
          best_dist = dist;
          best_cluster = c;
        }
      }
      if (labels[i] != best_cluster) {
        labels[i] = best_cluster;
        changed = true;
      }
    }
    // Update step
    Matrix<float> new_centroids(k_, nFeatures, 0.0f);
    std::vector<int> counts(k_, 0);
    for (int i = 0; i < nSamples; ++i) {
      int c = labels[i];
      for (int j = 0; j < nFeatures; ++j) {
        new_centroids.at(c, j) += X.at(i, j);
      }
      counts[c] += 1;
    }
    for (int c = 0; c < k_; ++c) {
      if (counts[c] > 0) {
        for (int j = 0; j < nFeatures; ++j) {
          new_centroids.at(c, j) /= counts[c];
        }
      } else {
        // Reinitialize empty cluster to a random point
        std::uniform_int_distribution<int> dis(0, nSamples - 1);
        int idx = dis(gen);
        for (int j = 0; j < nFeatures; ++j) {
          new_centroids.at(c, j) = X.at(idx, j);
        }
      }
    }
    centroids_ = new_centroids;
    if (!changed) {
      break; // Converged
    }
  }
  // Compute final inertia
  inertia_ = 0.0f;
  for (int i = 0; i < nSamples; ++i) {
    int c = labels[i];
    float dist = 0.0f;
    for (int j = 0; j < nFeatures; ++j) {
      float diff = X.at(i, j) - centroids_.at(c, j);
      dist += diff * diff;
    }
    inertia_ += dist;
  }
}

std::vector<int> models::KMeans::predict(const Matrix<float> &X) const
{
  const int nSamples = X.rows();
  const int nFeatures = X.cols();
  assert(centroids_.rows() == k_);
  assert(centroids_.cols() == nFeatures);
  std::vector<int> res;
  for (int i = 0; i < nSamples; ++i) {
    float best_dist = std::numeric_limits<float>::max();
    int best_cluster = -1;
    for (int c = 0; c < k_; ++c) {
      float dist = 0.0f;
      for (int j = 0; j < nFeatures; ++j) {
        float diff = X.at(i, j) - centroids_.at(c, j);
        dist += diff * diff;
      }
      if (dist < best_dist) {
        best_dist = dist;
        best_cluster = c;
      }
    }
    res.push_back(best_cluster);
  }
  return res;
}

const Matrix<float> &models::KMeans::centroids() const
{
  return centroids_;
}

float models::KMeans::inertia() const
{
  return inertia_;
}
