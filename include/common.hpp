#pragma once

#include "matrix.hpp"

#include <stdexcept>
#include <functional>
#include <vector>
#include <filesystem>
#include <string>
#include <random>
#include <cmath>


namespace utils {

  struct StepBreaker {
    void setStepCallback(std::function<bool()> callback) {
      stepf_ = callback;
    }
  protected:
    std::function<bool()> stepf_;
  };

  template <class T>
  struct Point {
    T x, y;
    Point(T x = 0, T y = 0) : x(x), y(y) {}
    auto operator<=>(const Point &other) const = default;
    // Optional: equality operator (not strictly necessary with <=> if defaulted)
    bool operator==(const Point &other) const = default;
  };
  using Pointf = Point<float>;

  inline std::string userDir() {
#ifdef _WIN32
    auto userDir = std::filesystem::path(std::getenv("USERPROFILE"));
#else
    auto userDir = std::filesystem::path(std::getenv("HOME")); // Linux/Mac
#endif
    return userDir.string();
  }

  // aka logistic function
  [[nodiscard]]
  inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
  }

  // Sigmoid derivative: s'(x) = s(x) * (1 - s(x)) 
  // @param sx - result of sigmoid(x)
  [[nodiscard]]
  inline float deriveSigmoid(float sx) {
    return sx * (1 - sx);
  }

  [[nodiscard]]
  inline float relu(float x) { return 0 < x ? x : 0; }

  [[nodiscard]]
  inline float leakyRelu(float x, float coef = 0.001f) { return 0 < x ? x : coef * x; }

  [[nodiscard]]
  inline float deriveReLU(float x) { return 0 < x ? 1 : 0; }

  [[nodiscard]]
  inline std::vector<float> softmax(const std::vector<float> &z) {
    if (z.empty()) return z;
    std::vector<float> result(z.size());
    // Find max for numerical stability
    float zmax = *std::max_element(z.begin(), z.end());
    float sum = 0.0f;
    for (size_t i = 0; i < z.size(); ++i) {
      result[i] = std::exp(z[i] - zmax);
      sum += result[i];
    }
    // Normalize to get probabilities
    if (sum > 0.0f) {
      for (float &val : result) {
        val /= sum;
      }
    } else {
      // Fallback: uniform distribution if all values are -infinity
      std::fill(result.begin(), result.end(), 1.0f / z.size());
    }
    return result;
  }


  [[nodiscard]]
  inline std::vector<std::vector<std::vector<std::vector<float>>>> 
    createRandomWeightsForConvLayer(int out_channels, int in_channels, int kernel_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    std::vector<std::vector<std::vector<std::vector<float>>>> weights(out_channels);
    for (int out_ch = 0; out_ch < out_channels; ++out_ch) {
      weights[out_ch].resize(in_channels);
      for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        weights[out_ch][in_ch].resize(kernel_size);
        for (int h = 0; h < kernel_size; ++h) {
          weights[out_ch][in_ch][h].resize(kernel_size);
          for (int w = 0; w < kernel_size; ++w) {
            weights[out_ch][in_ch][h][w] = dis(gen);
          }
        }
      }
    }
    return weights;
  }

  [[nodiscard]]
  inline std::vector<std::vector<std::vector<std::vector<float>>>>
    createXavierWeightsForConvLayer(int out_channels, int in_channels, int kernel_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // Xavier initialization: std = sqrt(2 / (fan_in + fan_out))
    int fan_in = in_channels * kernel_size * kernel_size;
    int fan_out = out_channels * kernel_size * kernel_size;
    float std_dev = std::sqrt(2.0f / (fan_in + fan_out));
    std::normal_distribution<float> dis(0.0f, std_dev);
    std::vector<std::vector<std::vector<std::vector<float>>>> weights(out_channels);
    for (int out_ch = 0; out_ch < out_channels; ++out_ch) {
      weights[out_ch].resize(in_channels);
      for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        weights[out_ch][in_ch].resize(kernel_size);
        for (int h = 0; h < kernel_size; ++h) {
          weights[out_ch][in_ch][h].resize(kernel_size);
          for (int w = 0; w < kernel_size; ++w) {
            weights[out_ch][in_ch][h][w] = dis(gen);
          }
        }
      }
    }
    return weights;
  }



} // namespace utils



namespace models {

  enum class ActivationF {
    Sigmoid,
    ReLU,
    Tanh,
    Softmax
  };

  enum class Initialization {
    RandomUniform,
    Xavier,
    He
  };

  class ActivationFunctions {
  public:
    [[nodiscard]]
    static std::vector<float> apply(const std::vector<float> &x, ActivationF af) {
      std::vector<float> result(x.size());
      switch (af) {
      case ActivationF::Sigmoid:
        for (size_t i = 0; i < x.size(); ++i) {
          result[i] = utils::sigmoid(x[i]);
        }
        break;
      case ActivationF::ReLU:
        for (size_t i = 0; i < x.size(); ++i) {
          result[i] = utils::relu(x[i]);
        }
        break;
      case ActivationF::Tanh:
        for (size_t i = 0; i < x.size(); ++i) {
          result[i] = std::tanh(x[i]);
        }
        break;
      case ActivationF::Softmax:
        result = utils::softmax(x);
        break;
      default:
        throw std::runtime_error("Unknown activation function");
      }
      return result;
    }
    [[nodiscard]]
    static std::vector<float> derivative(const std::vector<float> &x, ActivationF af) {
      std::vector<float> result(x.size());
      switch (af) {
      case ActivationF::Sigmoid:
        for (size_t i = 0; i < x.size(); ++i) {
          result[i] = utils::deriveSigmoid(utils::sigmoid(x[i]));
        }
        break;
      case ActivationF::ReLU:
        for (size_t i = 0; i < x.size(); ++i) {
          result[i] = x[i] > 0.0f ? 1.0f : 0.0f;
        }
        break;
      case ActivationF::Tanh:
        for (size_t i = 0; i < x.size(); ++i) {
          float t = std::tanh(x[i]);
          result[i] = 1.0f - t * t;
        }
        break;
      case ActivationF::Softmax:
        // For softmax, derivative depends on which output you're taking derivative w.r.t.
        // This returns diagonal elements of Jacobian (simplified case)
      {
        auto sm = apply(x, ActivationF::Softmax);
        for (size_t i = 0; i < x.size(); ++i) {
          result[i] = sm[i] * (1.0f - sm[i]);
        }
      }
      break;
      default:
        throw std::runtime_error("Unknown activation function");
      }
      return result;
    }
    [[nodiscard]]
    static float apply_single(float x, ActivationF af) {
      switch (af) {
      case ActivationF::Sigmoid:
        return utils::sigmoid(x);
      case ActivationF::ReLU:
        return utils::relu(x);
      case ActivationF::Tanh:
        return std::tanh(x);
      case ActivationF::Softmax:
        throw std::runtime_error("Softmax requires vector input");
      default:
        throw std::runtime_error("Unknown activation function");
      }
    }
    [[nodiscard]]
    static float derivative_single(float x, ActivationF af) {
      switch (af) {
      case ActivationF::Sigmoid:
        return utils::deriveSigmoid(x);
      case ActivationF::ReLU:
        return x > 0.0f ? 1.0f : 0.0f;
      case ActivationF::Tanh: {
        float t = std::tanh(x);
        return 1.0f - t * t;
      }
      case ActivationF::Softmax:
        throw std::runtime_error("Softmax derivative requires vector input");
      default:
        throw std::runtime_error("Unknown activation function");
      }
    }
  };

  class WeightInitializer {
  public:
    static void initialize_matrix(Matrix<float> &weights, int rows, int cols, Initialization method) {
      weights.resize(rows, cols);
      switch (method) {
      case Initialization::RandomUniform:
      {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.5, 0.5); // Match your MLP
        for (int j = 0; j < rows; ++j) {
          for (int k = 0; k < cols; ++k) {
            weights(j, k) = dis(gen);
          }
        }
      }
      break;
      case Initialization::Xavier:
      {
        float scale = std::sqrt(2.0f / (cols + rows)); // fan_in + fan_out
        for (int j = 0; j < rows; ++j) {
          for (int k = 0; k < cols; ++k) {
            weights(j, k) = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale;
          }
        }
      }
      break;
      case Initialization::He:
      {
        float scale = std::sqrt(2.0f / cols); // fan_in only
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);
        for (int j = 0; j < rows; ++j) {
          for (int k = 0; k < cols; ++k) {
            weights(j, k) = dist(gen);
          }
        }
      }
      break;
      default:
        throw std::runtime_error("Unknown initialization method");
      }
    }
#if 0
    static void initialize_matrix(std::vector<std::vector<float>> &weights, int rows, int cols, Initialization method) {
      weights.resize(rows, std::vector<float>(cols));
      switch (method) {
      case Initialization::RandomUniform:
      {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.5, 0.5); // Match your MLP
        for (auto &weight_row : weights) {
          for (auto &w : weight_row) {
            w = dis(gen);
          }
        }
      }
      break;
      case Initialization::Xavier:
      {
        float scale = std::sqrt(2.0f / (cols + rows)); // fan_in + fan_out
        for (int j = 0; j < rows; ++j) {
          for (int k = 0; k < cols; ++k) {
            weights[j][k] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale;
          }
        }
      }
      break;
      case Initialization::He:
      {
        float scale = std::sqrt(2.0f / cols); // fan_in only
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);
        for (auto &weight_row : weights) {
          for (auto &w : weight_row) {
            w = dist(gen);
          }
        }
      }
      break;
      default:
        throw std::runtime_error("Unknown initialization method");
      }
    }
#endif
    static void initialize_vector(std::vector<float> &vec, int size, Initialization method) {
      vec.resize(size);
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dist(-0.5, 0.5);
      const float scale = std::sqrt(2.0f / size); // fan_in + fan_out (approx)
      switch (method) {
      case Initialization::RandomUniform:
        for (auto &v : vec) v = dist(gen);
        break;
      case Initialization::Xavier:
        for (auto &v : vec) v = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale;
        break;
      case Initialization::He:
        if (1) {
          std::normal_distribution<float> dist(0.0f, scale);
          for (auto &v : vec) v = dist(gen);
        }
        break;
      }
    }

  private:
    //static void random_uniform_init(Matrix<float> &mat, float min_val, float max_val) {
    //  std::random_device rd;
    //  std::mt19937 gen(rd());
    //  std::uniform_real_distribution<float> dist(min_val, max_val);

    //  for (size_t i = 0; i < mat.rows(); ++i) {
    //    for (size_t j = 0; j < mat.cols(); ++j) {
    //      mat(i, j) = dist(gen);
    //    }
    //  }
    //}
    //static void xavier_init(Matrix<float> &mat) {
    //  // Match your MLP's Xavier implementation exactly
    //  float fan_in = static_cast<float>(mat.cols());
    //  float fan_out = static_cast<float>(mat.rows());
    //  float scale = std::sqrt(2.0f / (fan_in + fan_out));

    //  for (size_t i = 0; i < mat.rows(); ++i) {
    //    for (size_t j = 0; j < mat.cols(); ++j) {
    //      mat(i, j) = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale;
    //    }
    //  }
    //}
    //static void he_init(Matrix<float> &mat) {
    //  std::random_device rd;
    //  std::mt19937 gen(rd());
    //  float fan_in = static_cast<float>(mat.cols());
    //  float scale = std::sqrt(2.0f / fan_in);
    //  std::normal_distribution<float> dist(0.0f, scale);

    //  for (size_t i = 0; i < mat.rows(); ++i) {
    //    for (size_t j = 0; j < mat.cols(); ++j) {
    //      mat(i, j) = dist(gen);
    //    }
    //  }
    //}
  };
}
