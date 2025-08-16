#include <iostream>
#include <cassert>
#include <iomanip>

#include "matrix.hpp"
#include "models/cnn.hpp"

#define PRINT_MATRIX 0

namespace {
  void print_matrix(const Matrix<float> &mat, const std::string &name) {
#if PRINT_MATRIX
    std::cout << name << " (" << mat.rows() << "x" << mat.cols() << "):\n";
    for (size_t i = 0; i < mat.rows(); ++i) {
      for (size_t j = 0; j < mat.cols(); ++j) {
        std::cout << std::setw(8) << std::fixed << std::setprecision(3) << mat(i, j) << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
#endif
  }

  void print_result(const std::vector<float> &result, size_t width, size_t height, const std::string &name) {
#if PRINT_MATRIX
    std::cout << name << " (" << height << "x" << width << "):\n";
    for (size_t i = 0; i < height; ++i) {
      for (size_t j = 0; j < width; ++j) {
        std::cout << std::setw(8) << std::fixed << std::setprecision(3) << result[i * width + j] << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
#endif
  }

  bool compare_results(const std::vector<float> &actual, const std::vector<float> &expected, float tolerance = 1e-5f) {
    if (actual.size() != expected.size()) {
      return false;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
      if (std::abs(actual[i] - expected[i]) > tolerance) {
        return false;
      }
    }
    return true;
  }

  void print_test_result(const std::string &test_name, bool passed,
    const std::vector<float> &actual = {},
    const std::vector<float> &expected = {},
    size_t width = 0, size_t height = 0) {
    if (passed) {
      print_result(actual, width, height, "Result");
      std::cout << test_name << ": PASSED\n";
    } else {
      std::cout << test_name << ": FAILED\n";
      if (!actual.empty() && !expected.empty()) {
        print_result(actual, width, height, "Actual");
        print_result(expected, width, height, "Expected");
      }
    }
#if PRINT_MATRIX
    std::cout << "\n";
#endif
  }

} // anonymous namespace


void test_conv() {

  using namespace models;
  //Conv2D conv(3, 2, 3); // 3 input channels, 2 output channels, 3x3 kernel
  //conv.setWeights({
  //  {{{1, 0, -1}, {1, 0, -1}, {1, 0, -1}}, {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}}},
  //  {{{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}}, {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}}, {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}}}
  //  });



  auto out1 = utils::convolve1D(
    {3, 1, 4, 1, 5, 9},
    {7, 7, 5},
    1,
    0
  );

  // testing out1
  const std::vector<float> out1_test = { 3 * 7 + 1 * 7 + 4 * 5, 1 * 7 + 4 * 7 + 1 * 5, 4 * 7 + 1 * 7 + 5 * 5, 1 * 7 + 5 * 7 + 9 * 5 };

  assert(out1.size() == out1_test.size());

  if (out1_test == out1) {
    std::cout << "[utils::convolve1D] PASSED\n";
  } else {
    std::cout << "[utils::convolve1D] FAILED\n";
    for (const auto &val : out1) {
      std::cout << val << " ";
    }
  }
  std::cout << "\n";


  int passed = 0, total = 0;

  // Test 1: Basic 3x3 convolution with identity kernel
  {
    total++;
    std::string test_name = "[utils::convolve2D] Identity kernel           ";
    Matrix<float> input1 = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    Matrix<float> identity_kernel = {
        {0, 0, 0},
        {0, 1, 0},
        {0, 0, 0}
    };

    print_matrix(input1, "Input");
    print_matrix(identity_kernel, "Identity Kernel");

    auto result = utils::convolve2D(input1, identity_kernel, 1, 0);
    std::vector<float> expected = { 5 };

    bool test_passed = compare_results(result, expected);
    if (test_passed) passed++;
    print_test_result(test_name, test_passed, result, expected, 1, 1);
  }

  // Test 2: Simple edge detection
  {
    total++;
    std::string test_name = "[utils::convolve2D] Simple edge detection     ";
    Matrix<float> input2 = {
        {1, 1, 0},
        {1, 1, 0},
        {1, 1, 0}
    };
    Matrix<float> edge_kernel = {
        {-1, 1}
    };

    print_matrix(input2, "Input");
    print_matrix(edge_kernel, "Edge Kernel");

    auto result = utils::convolve2D(input2, edge_kernel, 1, 0);
    std::vector<float> expected = { 0, -1, 0, -1, 0, -1 };

    bool test_passed = compare_results(result, expected);
    if (test_passed) passed++;
    print_test_result(test_name, test_passed, result, expected, 2, 3);
  }

  // Test 3: 2x2 averaging kernel
  {
    total++;
    std::string test_name = "[utils::convolve2D] 2x2 averaging             ";
    Matrix<float> input3 = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    Matrix<float> avg_kernel = {
        {0.25f, 0.25f},
        {0.25f, 0.25f}
    };

    print_matrix(input3, "Input");
    print_matrix(avg_kernel, "Averaging Kernel");

    auto result = utils::convolve2D(input3, avg_kernel, 1, 0);
    std::vector<float> expected = { 3, 4, 6, 7 };  // (1+2+4+5)/4=3, (2+3+5+6)/4=4, etc.

    bool test_passed = compare_results(result, expected);
    if (test_passed) passed++;
    print_test_result(test_name, test_passed, result, expected, 2, 2);
  }

  // Test 4: Convolution with padding
  {
    total++;
    std::string test_name = "[utils::convolve2D] Convolution with padding=1";
    Matrix<float> input4 = {
        {5}
    };
    Matrix<float> kernel4 = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    print_matrix(input4, "Input (1x1)");
    print_matrix(kernel4, "3x3 Kernel");

    auto result = utils::convolve2D(input4, kernel4, 1, 1);
    // With padding=1, the 1x1 input becomes 3x3 padded with zeros
    // Only center gets the full kernel * 5 = 4*5 = 20
    std::vector<float> expected = { 20 };

    bool test_passed = compare_results(result, expected);
    if (test_passed) passed++;
    print_test_result(test_name, test_passed, result, expected, 1, 1);
  }

  // Test 5: Stride = 2 convolution
  {
    total++;
    std::string test_name = "[utils::convolve2D] Convolution with stride=2 ";
    Matrix<float> input5 = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };
    Matrix<float> kernel5 = {
        {1, 0},
        {0, 1}
    };

    print_matrix(input5, "Input (4x4)");
    print_matrix(kernel5, "2x2 diagonal kernel");

    auto result = utils::convolve2D(input5, kernel5, 2, 0);
    // stride=2: sample at (0,0), (0,2), (2,0), (2,2)
    // (0,0): 1*1 + 6*1 = 7
    // (0,2): 3*1 + 8*1 = 11
    // But wait, with stride=2 and 4x4 input, 2x2 kernel:
    // output size = (4-2)/2 + 1 = 2
    // So we get 2x2 output
    std::vector<float> expected = { 7, 11 };  // Only 1x2 output possible
    size_t output_width = (4 - 2) / 2 + 1;  // 2
    size_t output_height = (4 - 2) / 2 + 1; // 2
    // Actually it should be 2x2, let me recalculate:
    // (0,0): input[0][0]*kernel[0][0] + input[0][1]*kernel[0][1] + input[1][0]*kernel[1][0] + input[1][1]*kernel[1][1]
    //      = 1*1 + 2*0 + 5*0 + 6*1 = 7
    // (0,2): 3*1 + 4*0 + 7*0 + 8*1 = 11
    // (2,0): 9*1 + 10*0 + 13*0 + 14*1 = 23
    // (2,2): 11*1 + 12*0 + 15*0 + 16*1 = 27
    expected = { 7, 11, 23, 27 };

    bool test_passed = compare_results(result, expected);
    if (test_passed) passed++;
    print_test_result(test_name, test_passed, result, expected, output_width, output_height);
  }

  // Test 6: Single element convolution
  {
    total++;
    std::string test_name = "[utils::convolve2D] Single element            ";
    Matrix<float> input6 = { {7} };
    Matrix<float> kernel6 = { {3} };

    print_matrix(input6, "Input (1x1)");
    print_matrix(kernel6, "Kernel (1x1)");

    auto result = utils::convolve2D(input6, kernel6, 1, 0);
    std::vector<float> expected = { 21 };  // 7 * 3 = 21

    bool test_passed = compare_results(result, expected);
    if (test_passed) passed++;
    print_test_result(test_name, test_passed, result, expected, 1, 1);
  }

  // Test 7: Zero input
  {
    total++;
    std::string test_name = "[utils::convolve2D] Zero input                ";
    Matrix<float> input7 = Matrix<float>::zeros(3, 3);
    Matrix<float> kernel7 = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    print_matrix(input7, "Zero Input");
    print_matrix(kernel7, "Kernel");

    auto result = utils::convolve2D(input7, kernel7, 1, 0);
    std::vector<float> expected = { 0 };  // Zero input gives zero output

    bool test_passed = compare_results(result, expected);
    if (test_passed) passed++;
    print_test_result(test_name, test_passed, result, expected, 1, 1);
  }

  {
    total++;
    std::string test_name = "[utils::convolve2D] Vertical edge detection   ";
    Matrix<float> input8 = {
        {0, 1, 0},
        {0, 1, 0},
        {0, 1, 0}
    };
    Matrix<float> kernel8 = {
        {-1, 2, -1}
    };

    print_matrix(input8, "Input (vertical line)");
    print_matrix(kernel8, "Horizontal edge kernel");

    auto result = utils::convolve2D(input8, kernel8, 1, 0);
    // For each row: -1*0 + 2*1 + -1*0 = 2
    std::vector<float> expected = { 2, 2, 2 };

    bool test_passed = compare_results(result, expected);
    if (test_passed) passed++;
    print_test_result(test_name, test_passed, result, expected, 1, 3);
  }

  // Test 9: Cross kernel
  {
    total++;
    std::string test_name = "[utils::convolve2D] Cross kernel              ";
    Matrix<float> input9 = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    Matrix<float> cross_kernel = {
        {0, 1, 0},
        {1, -4, 1},
        {0, 1, 0}
    };

    print_matrix(input9, "Input");
    print_matrix(cross_kernel, "Cross Kernel (Laplacian-like)");

    auto result = utils::convolve2D(input9, cross_kernel, 1, 0);
    // Center calculation: 0*1 + 1*2 + 0*3 + 1*4 + (-4)*5 + 1*6 + 0*7 + 1*8 + 0*9
    //                   = 0 + 2 + 0 + 4 + (-20) + 6 + 0 + 8 + 0 = 0
    std::vector<float> expected = { 0 };

    bool test_passed = compare_results(result, expected);
    if (test_passed) passed++;
    print_test_result(test_name, test_passed, result, expected, 1, 1);
  }

  // Test 10: Rectangle input with small kernel
  {
    total++;
    std::string test_name = "[utils::convolve2D] Rectangle input           ";
    Matrix<float> input10 = {
        {1, 2, 3, 4},
        {5, 6, 7, 8}
    };
    Matrix<float> kernel10 = {
        {1, -1},
        {1, -1}
    };

    print_matrix(input10, "Input (2x4)");
    print_matrix(kernel10, "2x2 kernel");

    auto result = utils::convolve2D(input10, kernel10, 1, 0);
    // Output size: (2-2+1) x (4-2+1) = 1x3
    // (0,0): 1*1 + 2*(-1) + 5*1 + 6*(-1) = 1 - 2 + 5 - 6 = -2
    // (0,1): 2*1 + 3*(-1) + 6*1 + 7*(-1) = 2 - 3 + 6 - 7 = -2
    // (0,2): 3*1 + 4*(-1) + 7*1 + 8*(-1) = 3 - 4 + 7 - 8 = -2
    std::vector<float> expected = { -2, -2, -2 };

    bool test_passed = compare_results(result, expected);
    if (test_passed) passed++;
    print_test_result(test_name, test_passed, result, expected, 3, 1);
  }
}