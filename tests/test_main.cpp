#include <iostream>

extern void test_linear_regression();
extern void test_logistic_regression();
extern void test_knn_classification();
extern void test_naivebayes_classification();
extern void test_decision_tree();
extern void test_random_forest();
extern void test_svm();
extern void test_matrix();

int main() {
  std::cout << "Running basic tests..." << std::endl;

  test_linear_regression();
  test_logistic_regression();
  test_knn_classification();
  test_naivebayes_classification();
  test_decision_tree();
  test_random_forest();
  test_svm();
  test_matrix();

  return 0;
}
