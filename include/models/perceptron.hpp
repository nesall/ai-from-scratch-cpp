#include <vector>

namespace models {

  class Perceptron {
    float learningRate_ = 0.01f;
    size_t nofInputs_ = 0;
    std::vector<float> weights_;
  public:
    Perceptron(size_t input_size, float lr = 0.1f);
    void fit(const std::vector<std::vector<float>> &X, const std::vector<int> &y, size_t epochs = 10);
    int predict(const std::vector<float> &x);
  };

}