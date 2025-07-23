#include <vector>
#include <unordered_map>

namespace models {

  class GaussianNaiveBayes {
  public:
    void fit(const std::vector<float> &x_train, const std::vector<int> &y_train);
    int predict(float x_test);
    std::vector<int> predict(const std::vector<float> &x_tests);

  private:
    struct ClassStats {
      float mean = 0;
      float variance = 0;
      float prior = 0;
      int nof = 0;
    };

    std::unordered_map<int, ClassStats> model_;
  };


}