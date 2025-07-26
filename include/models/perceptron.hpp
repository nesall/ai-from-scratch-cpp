#include <vector>

namespace models {

  class Perceptron {
    float learningRate_ = 0.01f;
    size_t epochs_ = 0;
    size_t nofInputs_ = 0;
    std::vector<float> weights_;
  public:
    Perceptron(size_t input_size, float lr = 0.1f, size_t epochs = 10);
    void fit(const std::vector<std::vector<float>> &X, const std::vector<int> &y);
    int predict(const std::vector<float> &x);
  };

}