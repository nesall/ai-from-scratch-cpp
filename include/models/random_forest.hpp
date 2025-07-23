#include <vector>
#include "decision_tree.hpp"

namespace models {

  class RandomForest {
  public:
    RandomForest(int ntrees);

    void fit(const std::vector<float> &x_train, const std::vector<int> &y_train);
    int predict(float x) const;
    std::vector<int> predict(const std::vector<float> &x_tests) const;

  private:
    int nofTrees_ = 0;
    std::vector<DecisionTree> trees_;

    void bootstrap_sample(const std::vector<float> &x, const std::vector<int> &y,
      std::vector<float> &x_sample, std::vector<int> &y_sample) const;
    int majority_vote(const std::vector<int> &predictions) const;
  };

}