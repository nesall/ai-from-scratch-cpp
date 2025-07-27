#include <iostream>
#include <chrono>
#include <iomanip>

extern void test_linear_regression();
extern void test_logistic_regression();
extern void test_knn_classification();
extern void test_naivebayes_classification();
extern void test_decision_tree();
extern void test_random_forest();
extern void test_svm();
extern void test_matrix();
extern void test_perceptron();
extern void test_mlp();
extern void test_mnist_mlp();

int main() {
  std::cout << "Running basic tests..." << std::endl;

  auto run = [](auto &&func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << std::fixed << std::setprecision(4) << elapsed.count() << " secs\n\n";
    };

  run(test_linear_regression);
  run(test_logistic_regression);
  run(test_knn_classification);
  run(test_naivebayes_classification);
  run(test_decision_tree);
  run(test_random_forest);
  run(test_svm);
  run(test_matrix);
  run(test_perceptron);
  run(test_mlp);
  run(test_mnist_mlp);

  return 0;
}
