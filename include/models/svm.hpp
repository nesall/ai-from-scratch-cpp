#include <vector>

namespace models {

  class SupportVectorMachine {
    float learningRate_ = 0;
    float epochs_ = 0;
    float C_ = 0;
    std::vector<float> weights_;
    float bias_ = 0;
  public:
    SupportVectorMachine(float lr = 0.01f, int epochs = 1000, float C = 1.0f);
    void fit(const std::vector<std::vector<float>> &X, const std::vector<int> &y);
    int predict(const std::vector<float> &x) const;
  };

}