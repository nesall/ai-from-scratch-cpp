#include <iostream>

extern int test_linear_regression();

int main() {
  std::cout << "Running basic tests..." << std::endl;

  test_linear_regression();

  return 0;
}
