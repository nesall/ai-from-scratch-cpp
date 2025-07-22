#include <vector>

namespace models {

  class LinearRegression {
  private:
    float weight_ = 0;
    float bias_ = 0;
    float learning_rate_ = 0;
    int epochs_ = 0;

  public:
    LinearRegression(float lr = 0.01f, int ep = 1000);

    void fit(const std::vector<float> &x, const std::vector<float> &y);
    float predict(float x);
    std::vector<float> predict(const std::vector<float> &x_vals);
  };

}