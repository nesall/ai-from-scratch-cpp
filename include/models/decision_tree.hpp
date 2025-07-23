#include <vector>

namespace models {

  class DecisionTree {
  public:
    void fit(const std::vector<float> &x_train, const std::vector<int> &y_train);
    int predict(float x) const;
    std::vector<int> predict(const std::vector<float> &x_tests) const;

  private:
    struct Node {
      bool is_leaf;
      int class_label;
      float threshold;
      Node *left;
      Node *right;

      Node() : is_leaf(false), class_label(-1), threshold(0.0f), left(nullptr), right(nullptr) {}
    };

    Node *root_ = nullptr;

    Node *build_tree(const std::vector<float> &x, const std::vector<int> &y);
    int majority_class(const std::vector<int> &y) const;
    float compute_entropy(const std::vector<int> &y) const;
    float best_split(const std::vector<float> &x, const std::vector<int> &y, float &best_threshold) const;
    int predict_recursive(Node *node, float x) const;
    void free_tree(Node *node);
  };


}