#include <vector>

namespace models {

  class MLP {
  public:
    enum class ActivationF {
      Sigmoid,
      ReLU,
      Softmax
    };

    enum class Initialization {
      RandomUniform,
      Xavier,
      He
    };

  private:
    float learningRate_ = 0;
    std::vector<size_t> layerSizes_;

    struct Layer {
      std::vector<std::vector<float>> weights;  // [neurons][inputs_from_prev_layer]
      std::vector<float> biases;                // [neurons]
      std::vector<float> activations;           // [neurons] - cached during forward pass
      std::vector<float> deltas;                // [neurons] - for backprop
    };

    std::vector<Layer> layers_;
    ActivationF af_ = ActivationF::Sigmoid;

    //size_t numThreads_ = 1;

  public:
    MLP(const std::vector<size_t> &layers, float learning_rate, Initialization ini = Initialization::RandomUniform);
    void fit(const std::vector<std::vector<float>> &X, const std::vector<std::vector<float>> &y, size_t epochs, ActivationF af = ActivationF::Sigmoid);
    std::vector<float> predict(const std::vector<float> &x);

  private:
    void forwardPass(const std::vector<float> &input);
    void backwardPass(const std::vector<float> &input, const std::vector<int> &target);
    void updateWeights(const std::vector<float> &input);
  };

}