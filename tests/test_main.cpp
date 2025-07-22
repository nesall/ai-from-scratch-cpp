#include <iostream>

extern void test_linear_regression();
extern void test_logistic_regression();
extern void test_knn_classification();

int main() {
  std::cout << "Running basic tests..." << std::endl;

  test_linear_regression();
  test_logistic_regression();
  test_knn_classification();

  return 0;
}
