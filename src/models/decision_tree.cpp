#include "models/decision_tree.hpp"
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <cassert>
#include <numeric>

namespace {

  float compute_entropy(const std::unordered_map<int, int> &class_counts, int nofLabels = -1)
  {
    if (class_counts.empty()) return 0;
    if (nofLabels < 0) {
      nofLabels = 0;
      for (const auto &pair : class_counts) {
        nofLabels += pair.second;
      }
    }
    float entropy = 0.0f;
    for (const auto &pair : class_counts) {
      float p = pair.second / float(nofLabels);
      if (0 < p) {
        entropy -= p * std::log2(p);
      }
    }
    return entropy;
  }

}


void models::DecisionTree::fit(const std::vector<float> &x_train, const std::vector<int> &y_train)
{
  root_ = build_tree(x_train, y_train);
}

int models::DecisionTree::predict(float x) const
{
  if (!root_) {
    return -1;
  }
  return predict_recursive(root_, x);
}

std::vector<int> models::DecisionTree::predict(const std::vector<float> &x_tests) const
{
  std::vector<int> res;
  std::transform(x_tests.cbegin(), x_tests.cend(), std::back_inserter(res), [this](float x) { return predict(x); });
  return res;
}

models::DecisionTree::Node *models::DecisionTree::build_tree(const std::vector<float> &x, const std::vector<int> &y)
{
  if (x.empty() || y.empty() || x.size() != y.size()) {
    return nullptr;
  }
  Node *node = new Node();
  if (std::all_of(y.cbegin(), y.cend(), [y](int label) { return label == y[0]; })) {
    node->is_leaf = true;
    node->class_label = y[0];
    return node;
  }
  float best_threshold;
  float gain = best_split(x, y, best_threshold);
  if (gain <= 0.0f) {
    node->is_leaf = true;
    node->class_label = majority_class(y);
    return node;
  }
  node->threshold = best_threshold;
  std::vector<float> left_x, right_x;
  std::vector<int> left_y, right_y;
  for (size_t i = 0; i < x.size(); ++i) {
    if (x[i] <= best_threshold) {
      left_x.push_back(x[i]);
      left_y.push_back(y[i]);
    } else {
      right_x.push_back(x[i]);
      right_y.push_back(y[i]);
    }
  }
  node->left = build_tree(left_x, left_y);
  node->right = build_tree(right_x, right_y);
  if (!node->left || !node->right) {
    node->is_leaf = true;
    node->class_label = majority_class(y);
  }
  return node;
}

int models::DecisionTree::majority_class(const std::vector<int> &y) const
{
  if (y.empty()) return -1;
  std::unordered_map<int, int> class_counts;
  for (int label : y) {
    class_counts[label]++;
  }
  int majority_class = y[0];
  int max_count = 0;
  for (const auto &pair : class_counts) {
    if (pair.second > max_count) {
      max_count = pair.second;
      majority_class = pair.first;
    }
  }
  if (max_count == 0) return -1; // No valid majority class found
  if (max_count == 1) return y[0]; // If all classes are unique, return the first one
  if (max_count > 1) return majority_class; // Return the class with the highest count
  assert(!"should not reach here.");
  return 0;
}

float models::DecisionTree::compute_entropy(const std::vector<int> &labels) const
{
  if (labels.empty()) return 0;
  std::unordered_map<int, int> class_counts;
  for (int label : labels) class_counts[label]++;
  return ::compute_entropy(class_counts, labels.size());
}

float models::DecisionTree::best_split(const std::vector<float> &x, const std::vector<int> &y, float &best_threshold) const
{
  if (x.empty() || y.empty() || x.size() != y.size()) {
    return 0.0f;
  }
  float best_gain = 0.0f;
  best_threshold = 0.0f;
  float parent_entropy = compute_entropy(y);

  std::vector<std::pair<float, int>> combined;
  for (size_t i = 0; i < x.size(); ++i) {
    combined.emplace_back(x[i], y[i]);
  }
  std::sort(combined.begin(), combined.end());
  std::vector<float> sorted_x;

  std::vector<int> sorted_y;
  for (const auto &pair : combined) {
    sorted_x.push_back(pair.first);
    sorted_y.push_back(pair.second);
  }
  std::unordered_map<int, int> left_counts;
  std::unordered_map<int, int> right_counts;
  for (int label : sorted_y) {
    right_counts[label]++;
  }
  int left_size = 0;
  int right_size = sorted_y.size();
  for (size_t i = 0; i < sorted_x.size() - 1; ++i) {
    int current_label = sorted_y[i];
    left_counts[current_label]++;
    left_size++;
    right_counts[current_label]--;
    right_size--;
    if (sorted_x[i] == sorted_x[i + 1]) continue;
    float left_entropy = ::compute_entropy(left_counts);
    float right_entropy = ::compute_entropy(right_counts);
    float gain = parent_entropy - (left_size / float(left_size + right_size)) * left_entropy -
      (right_size / float(left_size + right_size)) * right_entropy;
    if (gain > best_gain) {
      best_gain = gain;
      best_threshold = (sorted_x[i] + sorted_x[i + 1]) / 2.0f; // Midpoint
    }
  }
  if (best_gain > 0.0f) {
    return best_gain;
  }
  best_threshold = 0.0f;
  return 0.0f;
}

int models::DecisionTree::predict_recursive(Node *node, float x) const
{
  if (!node) return -1;
  if (node->is_leaf) {
    return node->class_label;
  }
  if (x <= node->threshold) {
    return predict_recursive(node->left, x);
  } else {
    return predict_recursive(node->right, x);
  }
  return 0;
}

void models::DecisionTree::free_tree(Node *node)
{
  if (!node) return;
  free_tree(node->left);
  free_tree(node->right);
  delete node;
}

