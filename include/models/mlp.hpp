#ifndef MLP_HPP
#define MLP_HPP

#include "common.hpp"
#include "models/optimizers.hpp"
#include "models/regularization.hpp"
#include <vector>
#include <memory>

namespace models {

  class MLP {
  private:
    float learningRate_ = 0;
    std::vector<size_t> layerSizes_;

    struct Layer {
      Matrix<float> weights;  // [neurons][inputs_from_prev_layer]
      std::vector<float> biases;                // [neurons]
      std::vector<float> activations;           // [neurons] - cached during forward pass
      std::vector<float> deltas;                // [neurons] - for backprop
    };

    std::vector<Layer> layers_;
    ActivationF af_ = ActivationF::Sigmoid;

    std::unique_ptr<optimizers::Optimizer> opt_;
    std::unique_ptr<regularization::RegBase> reg_;
    std::unique_ptr<regularization::Dropout> drp_;

  public:
    MLP(const std::vector<size_t> &layers, float learning_rate, Initialization ini = Initialization::RandomUniform);
    void fit(const std::vector<std::vector<float>> &X, const std::vector<std::vector<float>> &y, size_t epochs, 
      ActivationF af = ActivationF::Sigmoid, float validationRatio = 0.2f, size_t patience = 20);
    std::vector<float> predict(const std::vector<float> &x);

    void setOptimizer(std::unique_ptr<optimizers::Optimizer> p) { opt_ = std::move(p); }
    void setRegularization(std::unique_ptr<regularization::RegBase> p) { reg_ = std::move(p); }
    void setDropout(std::unique_ptr<regularization::Dropout> p) { drp_ = std::move(p); }

  private:
    void forwardPass(const std::vector<float> &input);
    void backwardPass(const std::vector<float> &input, const std::vector<int> &target);
    void updateWeights(const std::vector<float> &input);
  };

}

#endif // MLP_HPP