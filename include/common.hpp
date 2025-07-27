#pragma once

#include <functional>
#include <vector>

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
}