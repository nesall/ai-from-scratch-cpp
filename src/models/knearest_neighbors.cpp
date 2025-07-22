#include "models/knearest_neighbors.hpp"

#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <unordered_map>


using namespace models;

KNearestNeighbors::KNearestNeighbors(int k)
{
  if (k <= 0) {
    throw std::invalid_argument("Number of neighbors k must be positive.");
  }
  k_ = k;
}

void KNearestNeighbors::fit(const std::vector<float> &x_train, const std::vector<int> &y_train)
{
  if (x_train.size() != y_train.size()) {
    throw std::invalid_argument("Input x_train and labels y_train must have the same length.");
  }
  if (x_train.empty()) {
    throw std::invalid_argument("Input x_train cannot be empty.");
  }
  x_train_ = x_train;
  y_train_ = y_train;
}

int KNearestNeighbors::predict(float x_test)
{
  if (x_train_.empty() || y_train_.empty()) {
    throw std::runtime_error("Model has not been trained yet.");
  }
  // Create a vector of pairs (distance, label)
  std::vector<std::pair<float, int>> distances;
  for (size_t i = 0; i < x_train_.size(); ++i) {
    float distance = std::abs(x_train_[i] - x_test);
    distances.emplace_back(distance, y_train_[i]);
  }

  // Sort by distance
  std::sort(distances.begin(), distances.end(),
    [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
      return a.first < b.first;
    });

  // Count the labels of the k nearest neighbors
  std::unordered_map<int, int> label_count;
  for (int i = 0; i < k_ && i < static_cast<int>(distances.size()); ++i) {
    label_count[distances[i].second]++;
  }

  // Find the label with the maximum count
  int max_count = 0;
  int predicted_label = -1;
  for (const auto &pair : label_count) {
    if (pair.second > max_count) {
      max_count = pair.second;
      predicted_label = pair.first;
    }
  }

  if (predicted_label == -1) {
    throw std::runtime_error("No valid label found for prediction.");
  }

  return predicted_label;
}

std::vector<int> KNearestNeighbors::predict(const std::vector<float> &x_tests)
{
  std::vector<int> res;
  std::transform(x_tests.cbegin(), x_tests.cend(), std::back_inserter(res), [this](float x) { return predict(x); });
  return res;
}
