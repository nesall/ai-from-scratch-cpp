#pragma once

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

  inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
  }

  // Sigmoid derivative: s'(x) = s(x) * (1 - s(x))
  inline float sigmoidDerivative(float x) {
    float s = /*sigmoid*/(x);
    return s * (1 - s);
  }

  inline float relu(float x) { return x > 0 ? x : 0; }

  inline float reluDerivative(float x) { return x > 0 ? 1 : 0; }

  inline std::vector<float> softmax(const std::vector<float> &z) {
    if (z.empty()) return z;
    std::vector<float> exp_z(z.size());
    float zmax = *std::max_element(z.cbegin(), z.cend());
    float sum = 0.0f;
    for (size_t i = 0; i < z.size(); ++i) {
      exp_z[i] = std::exp(z[i] - zmax);
      sum += exp_z[i];
    }
    for (float &val : exp_z) val /= sum;
    return exp_z;
  }


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