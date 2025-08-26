#include <iostream>
#include "datasynth.hpp"
#include "models/optimizers.hpp"


void test_sgd() {
  std::vector<float> params = { 1.0f };
  std::vector<float> grads = { 0.5f };

  optimizers::SGD opt(0.1f); // learning rate = 0.1
  opt.update(params, grads);

  std::cout << "[optimizers::SGD] ";
  float expected = 1.0f - 0.1f * 0.5f; // 0.95
  if (nearly_equal(params[0], expected)) {
    std::cout << "PASSED \n";
  } else {
    std::cout << "FAILED (got " << params[0] << " expected " << expected << ")\n";
  }
}

void test_momentum() {
  std::vector<float> params = { 1.0f };
  std::vector<float> grads = { 0.5f };

  optimizers::Momentum opt(0.1f, 0.9f); // lr=0.1, beta=0.9
  opt.update(params, grads);

  std::cout << "[optimizers::Momentum] ";
  float expected = 1.0f - 0.1f * 0.5f; // first step same as SGD
  if (nearly_equal(params[0], expected)) {
    std::cout << "PASSED\n";
  } else {
    std::cout << "FAILED (got " << params[0] << " expected " << expected << ")\n";
  }
}

void test_rmsprop() {
  std::vector<float> params = { 1.0f };
  std::vector<float> grads = { 0.5f };
  optimizers::RMSProp opt(0.1f); // lr=0.1
  opt.update(params, grads);
  // RMSProp with beta=0.9, eps=1e-8:
  // v = 0.9*0 + 0.1*0.25 = 0.025
  // update = lr * grad / (sqrt(v) + eps) = 0.1 * 0.5 / (sqrt(0.025)+1e-8) = 0.1
  float update = 0.1f * 0.5f / (std::sqrt(0.25f * 0.1f) + 1e-8f);
  float expected = 1.0f - update;
  std::cout << "[optimizers::RMSProp] ";
  if (nearly_equal(params[0], expected, 1e-3f)) {
    std::cout << "PASSED\n";
  } else {
    std::cout << "FAILED (got " << params[0] << " expected ~" << expected << ")\n";
  }
}

void test_adam() {
  std::vector<float> params = { 1.0f };
  std::vector<float> grads = { 0.5f };

  optimizers::Adam opt(0.1f); // lr=0.1
  opt.update(params, grads);

  // Adam with beta1=0.9, beta2=0.999, t=1:
  // m_hat = grad / (1 - 0.9) = 0.5 / 0.1 = 5.0
  // v_hat = grad^2 / (1 - 0.999) = 0.25 / 0.001 = 250.0
  // update = lr * m_hat / (sqrt(v_hat) + eps)
  float update = 0.1f * 0.5f / (std::sqrt(0.25f) + 1e-8f);
  float expected = 1.0f - update;

  std::cout << "[optimizers::Adam] ";
  if (nearly_equal(params[0], expected, 1e-3f)) {
    std::cout << "PASSED\n";
  } else {
    std::cout << "FAILED (got " << params[0] << " expected ~" << expected << ")\n";
  }
#if 0 // this test passes ok.
  std::cout << "[optimizers::Adam] step-by-step test\n";

  std::vector<float> params = { 1.0f };
  std::vector<float> grads = { 0.5f };

  optimizers::Adam opt(0.1f); // lr=0.1, beta1=0.9, beta2=0.999

  // Step 1 (already validated)
  opt.update(params, grads);
  float expected1 = 0.9f; // from manual math
  if (nearly_equal(params[0], expected1, 1e-5f)) {
    std::cout << "Step 1: PASSED (" << params[0] << ")\n";
  } else {
    std::cout << "Step 1: FAILED (got " << params[0] << " expected " << expected1 << ")\n";
  }

  // Step 2: recalc manually
  // m2 = 0.9*0.05 + 0.1*0.5 = 0.095
  // v2 = 0.999*0.00025 + 0.001*0.25 = 0.00049975
  // m_hat = 0.095 / (1 - 0.9^2) = 0.095 / 0.19 = 0.5
  // v_hat = 0.00049975 / (1 - 0.999^2) = 0.00049975 / 0.001999 = 0.25
  // update = 0.1 * (0.5 / (sqrt(0.25)+eps)) = 0.1
  // param = 0.9 - 0.1 = 0.8
  opt.update(params, grads);
  float expected2 = 0.8f;
  if (nearly_equal(params[0], expected2, 1e-5f)) {
    std::cout << "Step 2: PASSED (" << params[0] << ")\n";
  } else {
    std::cout << "Step 2: FAILED (got " << params[0] << " expected " << expected2 << ")\n";
  }

  // Step 3: similar reasoning, should converge steadily by -0.1 each step
  opt.update(params, grads);
  float expected3 = 0.7f;
  if (nearly_equal(params[0], expected3, 1e-5f)) {
    std::cout << "Step 3: PASSED (" << params[0] << ")\n";
  } else {
    std::cout << "Step 3: FAILED (got " << params[0] << " expected " << expected3 << ")\n";
  }
#endif
}

void test_optimizers()
{
  test_sgd();
  test_momentum();
  test_rmsprop();
  test_adam();

}