#pragma once

#include <functional>

namespace commons {

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

}