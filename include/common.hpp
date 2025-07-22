#pragma once

#include <functional>
#include <compare>

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
    // Three-way comparison operator
    auto operator<=>(const Point &other) const = default;
    // Optional: equality operator (not strictly necessary with <=> if defaulted)
    bool operator==(const Point &other) const = default;
  };
  using Pointf = Point<float>;

}