#include <iostream>

extern void test_linear_regression();
extern void test_logistic_regression();
extern void test_knn_classification();
extern void test_naivebayes_classification();
extern void test_decision_tree();

int main() {
  std::cout << "Running basic tests..." << std::endl;

  test_linear_regression();
  test_logistic_regression();
  test_knn_classification();
  test_naivebayes_classification();
  test_decision_tree();

  return 0;
}
