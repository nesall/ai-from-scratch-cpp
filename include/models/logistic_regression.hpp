#pragma once

#include <vector>

namespace models {

  class LogisticRegression {
  private:
    float weight_ = 0;
    float bias_ = 0;
    float learning_rate_ = 0;
    int epochs_ = 0;

    float sigmoid(float z) const;

  public:
    LogisticRegression(float lr = 0.01f, int ep = 1000);
    void fit(const std::vector<float> &x, const std::vector<int> &y);
    float predict_proba(float x) const;
    int predict(float x) const;
    std::vector<int> predict(const std::vector<float> &x_vals) const;
  };

}