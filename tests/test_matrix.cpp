#include <iostream>
#include "matrix.hpp"
#include "datasynth.hpp"


bool test_addition() {
  Matrix A({ {1, 2}, {3, 4} });
  Matrix B({ {5, 6}, {7, 8} });
  Matrix C = A + B;

  return nearly_equal(C.at(0, 0), 6) &&
    nearly_equal(C.at(0, 1), 8) &&
    nearly_equal(C.at(1, 0), 10) &&
    nearly_equal(C.at(1, 1), 12);
}

bool test_subtraction() {
  Matrix A({ {5, 6}, {7, 8} });
  Matrix B({ {1, 2}, {3, 4} });
  Matrix C = A - B;

  return nearly_equal(C.at(0, 0), 4) &&
    nearly_equal(C.at(0, 1), 4) &&
    nearly_equal(C.at(1, 0), 4) &&
    nearly_equal(C.at(1, 1), 4);
}

bool test_scalar_multiplication() {
  Matrix A({ {1, 2}, {3, 4} });
  Matrix B = A * 2.0f;

  return nearly_equal(B.at(0, 0), 2) &&
    nearly_equal(B.at(0, 1), 4) &&
    nearly_equal(B.at(1, 0), 6) &&
    nearly_equal(B.at(1, 1), 8);
}

bool test_matrix_multiplication() {
  Matrix A({ {1, 2}, {3, 4} });
  Matrix B({ {2, 0}, {1, 2} });
  Matrix C = A * B;

  return nearly_equal(C.at(0, 0), 4) &&
    nearly_equal(C.at(0, 1), 4) &&
    nearly_equal(C.at(1, 0), 10) &&
    nearly_equal(C.at(1, 1), 8);
}

bool test_transpose() {
  Matrix A({ {1, 2, 3}, {4, 5, 6} });
  Matrix At = A.transpose();

  return At.rows() == 3 &&
    At.cols() == 2 &&
    nearly_equal(At.at(0, 0), 1) &&
    nearly_equal(At.at(1, 0), 2) &&
    nearly_equal(At.at(2, 0), 3) &&
    nearly_equal(At.at(0, 1), 4) &&
    nearly_equal(At.at(1, 1), 5) &&
    nearly_equal(At.at(2, 1), 6);
}

bool test_invert() {
  // Inversion is not implemented in the provided code, so this test will always fail.
   // You can implement the invert method in Matrix class and then uncomment this test.



  return false;
}

void test_matrix() {
  std::cout << "[Matrix] Addition:                " << (test_addition() ? "PASSED" : "FAILED") << std::endl;
  std::cout << "[Matrix] Subtraction:             " << (test_subtraction() ? "PASSED" : "FAILED") << std::endl;
  std::cout << "[Matrix] Scalar Multiplication:   " << (test_scalar_multiplication() ? "PASSED" : "FAILED") << std::endl;
  std::cout << "[Matrix] Matrix Multiplication:   " << (test_matrix_multiplication() ? "PASSED" : "FAILED") << std::endl;
  std::cout << "[Matrix] Transpose:               " << (test_transpose() ? "PASSED" : "FAILED") << "\n";
}
