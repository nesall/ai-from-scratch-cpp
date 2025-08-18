#include "mlp.hpp"
#include <vector>

namespace models {

  class Conv2D {
  public:
    Conv2D(int out_channels, int in_channels, int kernel_size, int stride = 1, int padding = 0);

    void setWeights(const std::vector<std::vector<std::vector<std::vector<float>>>> &weights);

    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>> &input);

  private:
    int out_channels_ = 0;
    int in_channels_ = 0;
    int kernel_size_ = 0;
    int stride_ = 0;
    int padding_ = 0;

    std::vector<std::vector<std::vector<std::vector<float>>>> weights_; // [out][in][H][W]
  };


  class MaxPool2D {
  public:
    MaxPool2D(int pool_size, int stride = -1);
    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>> &input);

  private:
    int pool_size_ = 0;
    int stride_ = 0;
  };


  class CNN {
  public:
    CNN(int in_channels, const std::vector<size_t> &layers, float learningRate);

    // Example: configure CNN with conv/pool layers and MLP for classification
    void add_conv_layer(int out_channels, int kernel_size, int stride = 1, int padding = 0);
    void add_pool_layer(int pool_size, int stride = -1);
    void set_conv_weights(size_t layer_index, const std::vector<std::vector<std::vector<std::vector<float>>>> &weights);

    std::vector<float> forward(const std::vector<std::vector<std::vector<float>>> &input);
    int predict_class(const std::vector<std::vector<std::vector<float>>> &input);

    size_t get_conv_layer_count() const { return conv_layers_.size(); }

    // Extract features without MLP classification (for training)
    std::vector<float> extract_features(const std::vector<std::vector<std::vector<float>>> &input);

    // Train the CNN (train MLP on extracted features)
    void train(const std::vector<std::vector<std::vector<std::vector<float>>>> &train_images,
      const std::vector<std::vector<float>> &train_labels,
      size_t epochs = 100,
      ActivationF activation = ActivationF::Softmax);

  private:
    int in_channels_ = 0;
    int current_channels_ = 0;
    std::vector<Conv2D> conv_layers_;
    std::vector<MaxPool2D> pool_layers_;
    MLP mlp_;

    std::vector<float> flatten(const std::vector<std::vector<std::vector<float>>> &input);
  };

}