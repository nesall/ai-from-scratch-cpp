#ifndef RNN_HPP
#define RNN_HPP

#include "common.hpp"
#include <vector>

namespace models {

  class RNN {
  public:
    RNN(int input_size, int hidden_size, int output_size, bool initW = true);

    void reset_state();

    void fit(const std::vector<std::vector<std::vector<float>>> &sequences,
      const std::vector<std::vector<std::vector<float>>> &targets,
      int epochs, float learning_rate,
      ActivationF hidden_af = ActivationF::Tanh,
      ActivationF output_af = ActivationF::Sigmoid);

    std::vector<std::vector<float>> predict(const std::vector<std::vector<float>> &sequence);

    void initialize_weights(Initialization method = Initialization::Xavier);

    // Optional: Save/Load weights (like your MLP might need)
    // void save_weights(const std::string& filename);
    // void load_weights(const std::string& filename);

  private:
    int input_size_;
    int hidden_size_;
    int output_size_;

    // Parameters
    Matrix<float> Wxh_; // input -> hidden
    Matrix<float> Whh_; // hidden -> hidden
    Matrix<float> Why_; // hidden -> output
    std::vector<float> bh_;
    std::vector<float> by_;

    // State
    std::vector<float> h_prev_; // previous hidden state [hidden_size]

    ActivationF current_hidden_af_ = ActivationF::Tanh;
    ActivationF current_output_af_ = ActivationF::Sigmoid;

    struct TimeStep {
      std::span<const float> input;              // input at this timestep
      std::vector<float> hidden_raw;         // hidden pre-activation (Wxh*x + Whh*h_prev + bh)
      std::vector<float> hidden;             // hidden activation (after tanh/sigmoid/etc.)
      std::vector<float> output_raw;         // output pre-activation (Why*h + by)
      std::vector<float> output;             // output activation (final prediction)
      std::vector<float> h_prev;             // previous hidden state (for this timestep)
    };
    std::vector<TimeStep> timesteps_;       // Store sequence for backprop through time

    // Gradients
    Matrix<float> dWxh_; // input -> hidden gradients
    Matrix<float> dWhh_; // hidden -> hidden gradients
    Matrix<float> dWhy_; // hidden -> output gradients
    std::vector<float> dbh_;
    std::vector<float> dby_;

  private:
    std::vector<std::vector<float>> forward(const Matrix<float> &sequence);

    // Loss computation
    float compute_loss(const Matrix<float> &predictions, const Matrix<float> &targets);

    // Single sequence training (for online learning)
    float train_sequence(const Matrix<float> &sequence, const Matrix<float> &targets, float learning_rate);

    // Forward pass for training (stores timesteps for backprop)
    Matrix<float> forward_training(const Matrix<float> &sequence);

    // Compute single timestep
    void forward_timestep(
      std::span<const float> input,
      const std::vector<float> &h_prev,
      std::vector<float> &hidden_raw,
      std::vector<float> &hidden,
      std::vector<float> &output_raw,
      std::vector<float> &output);

    // Backpropagation through time
    void backward_through_time(const Matrix<float> &targets);

    // Update weights using computed gradients
    void update_weights(float learning_rate);

    // Zero out gradients
    void zero_gradients();

    // Helpers
    std::vector<float> matvec_mul(const Matrix<float> &W, std::span<const float> x);
    std::vector<float> matvec_mul(const Matrix<float> &W, const std::vector<float> &x);
    std::vector<float> add(const std::vector<float> &a, const std::vector<float> &b);
    std::vector<float> tanh_vec(const std::vector<float> &x);

    std::vector<float> apply_activation(const std::vector<float> &x, ActivationF af);
    std::vector<float> activation_derivative(const std::vector<float> &x, ActivationF af);

    // Helper function to compute numerical gradients via finite differences
    static Matrix<float> numerical_gradient(
      models::RNN &rnn,
      const Matrix<float> &sequence,
      const Matrix<float> &targets,
      Matrix<float> &weight_matrix,
      float epsilon = 1e-4);

  public:
    static void run_basic_test();
  };

}

#endif // RNN_HPP
