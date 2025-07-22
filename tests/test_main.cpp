#include <iostream>

extern int test_linear_regression();
extern int test_logistic_regression();

int main() {
  std::cout << "Running basic tests..." << std::endl;

  test_linear_regression();
  test_logistic_regression();

  return 0;
}
