#include "models/random_forest.hpp"
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <cassert>


models::RandomForest::RandomForest(int ntrees)
{
  nofTrees_ = ntrees;
  trees_.reserve(ntrees);
  for (int i = 0; i < ntrees; i ++)
    trees_.emplace_back();
}

void models::RandomForest::fit(const std::vector<float> &x_train, const std::vector<int> &y_train)
{
  assert(x_train.size() == y_train.size());  
  for (int i = 0; i < nofTrees_; ++i) {
    std::vector<float> x_sample;
    std::vector<int> y_sample;
    bootstrap_sample(x_train, y_train, x_sample, y_sample);
    trees_[i].fit(x_sample, y_sample);
  }
}

int models::RandomForest::predict(float x) const
{
  assert(!trees_.empty());
  std::vector<int> predictions;
  predictions.reserve(trees_.size());
  for (const auto &tree : trees_) {
    predictions.push_back(tree.predict(x));
  }
  return majority_vote(predictions);
}

std::vector<int> models::RandomForest::predict(const std::vector<float> &x_tests) const
{
  std::vector<int> res;
  std::transform(x_tests.cbegin(), x_tests.cend(), std::back_inserter(res), [this](float x) { return predict(x); });
  return res;
}

void models::RandomForest::bootstrap_sample(const std::vector<float> &x, const std::vector<int> &y, std::vector<float> &x_sample, std::vector<int> &y_sample) const
{
  assert(x.size() == y.size());
  size_t n = x.size();
  x_sample.clear();
  y_sample.clear();
  for (size_t i = 0; i < n; ++i) {
    size_t idx = rand() % n; // Random index
    x_sample.push_back(x[idx]);
    y_sample.push_back(y[idx]);
  }
}

int models::RandomForest::majority_vote(const std::vector<int> &predictions) const
{
  assert(!predictions.empty());
  std::unordered_map<int, int> countMap;
  for (int label : predictions) {
    countMap[label]++;
  }
  int majorityLabel = -1;
  int maxCount = 0;
  for (const auto &pair : countMap) {
    if (pair.second > maxCount) {
      maxCount = pair.second;
      majorityLabel = pair.first;
    }
  }
  assert(majorityLabel != -1);  
  return majorityLabel;
}
