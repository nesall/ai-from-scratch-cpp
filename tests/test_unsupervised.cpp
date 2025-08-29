#include <iostream>
#include <vector>
#include <random>
#include "models/kmeans.hpp"
#include "models/pca.hpp"


// Generate synthetic 2D Gaussian clusters
Matrix<float> make_blobs(int n_samples, int n_clusters, float spread = 0.5f, int seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> center_dist(-5.0f, 5.0f);
  std::normal_distribution<float> noise(0.0f, spread);

  std::vector<std::pair<float, float>> centers;

  // pick random cluster centers
  for (int c = 0; c < n_clusters; ++c) {
    centers.emplace_back(center_dist(gen), center_dist(gen));
  }

  Matrix<float> data(n_samples, 2);
  // assign points around each center
  for (int i = 0; i < n_samples; ++i) {
    auto [cx, cy] = centers[i % n_clusters];
    float x = cx + noise(gen);
    float y = cy + noise(gen);
    data(i, 0) = x;
    data(i, 1) = y;
  }
  return data;
}

void test_kmeans() {
  int n_samples = 300;
  int k = 3;
  auto X = make_blobs(n_samples, k);

  models::KMeans km(k, 100, true);
  km.fit(X);

  //std::cout << "Final inertia (WCSS): " << km.inertia() << "\n";
  //std::cout << "Centroids:\n";
  //int idx = 0;
  //auto centroids = km.centroids();
  //for (auto c : centroids) {
  //  std::cout << "  " << idx++ << ": (" << c[0] << ", " << c[1] << ")\n";
  //}

  // Predict a new point
  std::vector<std::vector<float>> new_pts = { 
    {0.0f, 0.0f}, {-4.0f, 6.0f}, {3.0f, 2.0f},
    {3.0f, -0.5f}, {-5.0f, 4.0f}, {3.0f, 3.0f},
  };
  std::vector<int> expected{ 1, 0, 2, 1, 0, 2 }; // expected clusters for the above points
  auto preds = km.predict(new_pts);
  std::cout << "[k-Means] " << (expected == preds ? "PASSED" : "FAILED") << "\n";

  //std::cout << "Predicted cluster assignments for test points:\n";
  //for (size_t i = 0; i < new_pts.size(); ++i) {
  //  std::cout << "  (" << new_pts[i][0] << "," << new_pts[i][1] << ") -> cluster " << preds[i] << "\n";
  //}

  //std::vector<int> labels = km.predict(X);
  //std::vector<int> counts(k, 0);
  //for (int l : labels) counts[l]++;
  //std::cout << "Cluster sizes: ";
  //for (int i = 0; i < k; ++i) std::cout << counts[i] << " ";
  //std::cout << std::endl;
}

// Make synthetic 3D data with correlations
std::vector<std::vector<float>> make_correlated_data(int n_samples, int seed = 123) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> gauss(0.0f, 1.0f);

  std::vector<std::vector<float>> X;
  for (int i = 0; i < n_samples; ++i) {
    float z = gauss(gen);  // main latent factor
    float x = z + 0.1f * gauss(gen);
    float y = 0.5f * z + 0.1f * gauss(gen);
    float w = 0.2f * z + 0.1f * gauss(gen);
    X.push_back({ x, y, w });
  }
  return X;
}

void test_pca() {
  int n_samples = 200;
  auto X = make_correlated_data(n_samples);

  models::PCA pca(2);  // reduce to 2 dimensions
  pca.fit(X);

  auto X_reduced = pca.transform(X, 2);
  auto X_reconstructed = pca.inverseTransform(X_reduced);

  std::cout << "Explained variance per component:\n";
  auto var = pca.explainedVariance();
  for (size_t i = 0; i < var.size(); ++i) {
    std::cout << "  PC" << i + 1 << ": " << var[i] << "\n";
  }

  // Compute mean reconstruction error
  double err = 0.0;
  for (size_t i = 0; i < X.size(); ++i) {
    for (size_t j = 0; j < X[i].size(); ++j) {
      double diff = X[i][j] - X_reconstructed[i][j];
      err += diff * diff;
    }
  }
  err /= X.size();
  std::cout << "Mean squared reconstruction error: " << err << "\n";

  // Print first few reduced points
  std::cout << "First 5 reduced samples:\n";
  for (int i = 0; i < 5; ++i) {
    std::cout << "  (" << X_reduced[i][0] << ", " << X_reduced[i][1] << ")\n";
  }
}