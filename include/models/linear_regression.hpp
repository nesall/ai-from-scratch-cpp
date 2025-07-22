#include <vector>
#include "../include/common.hpp"

namespace models {

  class LinearRegression : public commons::StepBreaker {
  private:
    float weight_ = 0;
    float bias_ = 0;
    float learning_rate_ = 0;
    int epochs_ = 0;

  public:
    LinearRegression(float lr = 0.01f, int ep = 1000);

    void fit(const std::vector<commons::Pointf> &pts);
    float predict(float x);
    std::vector<float> predict(const std::vector<float> &x_vals);
  };

}