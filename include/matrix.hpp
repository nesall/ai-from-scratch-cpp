#pragma once

#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <cassert>
#include <span>
#include <algorithm>


template <typename T = float>
class Matrix {
public:
  using value_type = T;

  Matrix() = default;
  Matrix(Matrix &&) = default;
  Matrix(const Matrix &) = default;
  Matrix &operator = (const Matrix &) = default;
  Matrix &operator = (Matrix &&) = default;
  Matrix(size_t rows, size_t cols, const T &v = {})
    : rows_(rows), cols_(cols), data_(rows *cols, v) {}

  Matrix(std::initializer_list<std::initializer_list<T>> list) {
    rows_ = list.size();
    if (0 < rows_) {
      cols_ = list.begin()->size();
      data_.reserve(rows_ * cols_);
      for (const auto &row : list) {
        if (row.size() != cols_) {
          throw std::runtime_error("All rows must have the same number of columns");
        }
        data_.insert(data_.end(), row.begin(), row.end());
      }
    }
  }

  Matrix(const std::vector<std::vector<T>> &list) {
    rows_ = list.size();
    if (0 < rows_) {
      cols_ = list.begin()->size();
      data_.reserve(rows_ * cols_);
      for (const auto &row : list) {
        if (row.size() != cols_) {
          throw std::runtime_error("All rows must have the same number of columns");
        }
        data_.insert(data_.end(), row.begin(), row.end());
      }
    }
  }

  void resize(size_t rows, size_t cols, const T &v = {}) {
    rows_ = rows;
    cols_ = cols;
    data_.resize(rows * cols, v);
  }

  T &at(size_t row, size_t col) {
    check_bounds(row, col);
    return data_.at(index(row, col));
  }
  T at(size_t row, size_t col) const {
    check_bounds(row, col);
    return data_[index(row, col)];
  }

  Matrix operator+(const Matrix &other) const {
    check_same_shape(other);
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result(i, j) = (*this)(i, j) + other(i, j);
      }
    }
    return result;
  }
  Matrix operator-(const Matrix &other) const {
    check_same_shape(other);
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result(i, j) = (*this)(i, j) - other(i, j);
      }
    }
    return result;
  }
  Matrix operator*(const Matrix &other) const {
    assert(cols_ == other.rows_);
    Matrix result(rows_, other.cols_);
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < other.cols_; ++j) {
        for (size_t k = 0; k < cols_; k ++) {
          result(i, j) += (*this)(i, k) * other(k, j);
        }
      }
    }
    return result;
  }
  Matrix operator*(float scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result(i, j) = (*this)(i, j) * scalar;
      }
    }
    return result;
  }

  Matrix transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result(j, i) = (*this)(i, j);
      }
    }
    return result;
  }

  void fill(const T &v) {
    std::fill(data_.begin(), data_.end(), v);
  }
 
  static Matrix identity(size_t size) {
    Matrix result(size, size);
    for (size_t i = 0; i < size; ++i) {
      result.at(i, i) = 1;
    }
    return result;
  }

  static Matrix zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols);
  }

  static Matrix ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, T{ 1 });
  }

  T &operator()(size_t row, size_t col) {
    return at(row, col);
  }

  T operator()(size_t row, size_t col) const {
    return at(row, col);
  }

  std::span<T> operator[](size_t row) {
    check_bounds(row, 0);
    return std::span<T>(&data_[index(row, 0)], cols_);
  }

  std::span<const T> operator[](size_t row) const {
    check_bounds(row, 0);
    return std::span<const T>(&data_[index(row, 0)], cols_);
  }

  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t nofElems() const { return data_.size(); }
  bool empty() const { return data_.empty(); }

  // Iterator support for range-based for loops
  class row_iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = std::span<T>;
    using pointer = value_type *;
    using reference = value_type;

    row_iterator(T *data, size_t cols, size_t row_index)
      : data_(data), cols_(cols), row_index_(row_index) {
    }

    reference operator*() const {
      return std::span<T>(data_ + row_index_ * cols_, cols_);
    }

    row_iterator &operator++() {
      ++row_index_;
      return *this;
    }

    row_iterator operator++(int) {
      row_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const row_iterator &a, const row_iterator &b) {
      return a.row_index_ == b.row_index_;
    }

    friend bool operator!=(const row_iterator &a, const row_iterator &b) {
      return a.row_index_ != b.row_index_;
    }

  private:
    T *data_;
    size_t cols_;
    size_t row_index_;
  };

  class const_row_iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = std::span<const T>;
    using pointer = value_type *;
    using reference = value_type;

    const_row_iterator(const T *data, size_t cols, size_t row_index)
      : data_(data), cols_(cols), row_index_(row_index) {
    }

    reference operator*() const {
      return std::span<const T>(data_ + row_index_ * cols_, cols_);
    }

    const_row_iterator &operator++() {
      ++row_index_;
      return *this;
    }

    const_row_iterator operator++(int) {
      const_row_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const const_row_iterator &a, const const_row_iterator &b) {
      return a.row_index_ == b.row_index_;
    }

    friend bool operator!=(const const_row_iterator &a, const const_row_iterator &b) {
      return a.row_index_ != b.row_index_;
    }

  private:
    const T *data_;
    size_t cols_;
    size_t row_index_;
  };

  row_iterator begin() {
    return row_iterator(data_.data(), cols_, 0);
  }

  row_iterator end() {
    return row_iterator(data_.data(), cols_, rows_);
  }

  const_row_iterator begin() const {
    return const_row_iterator(data_.data(), cols_, 0);
  }

  const_row_iterator end() const {
    return const_row_iterator(data_.data(), cols_, rows_);
  }

  const_row_iterator cbegin() const {
    return const_row_iterator(data_.data(), cols_, 0);
  }

  const_row_iterator cend() const {
    return const_row_iterator(data_.data(), cols_, rows_);
  }

private:
  size_t rows_ = 0;
  size_t cols_ = 0;

  std::vector<T> data_;

  void check_bounds(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
      throw std::runtime_error("Matrix index out of bounds");
    }
  }

  void check_same_shape(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::runtime_error("Matrices must have the same shape");
    }
  }

  size_t index(size_t row, size_t col) const {
    return row * cols_ + col;
  }
};

class MatrixOps {
public:
  // Matrix-Vector operations (for your Matrix class)
  static std::vector<float> matvec_mul(const Matrix<float> &W, const std::vector<float> &x) {
    return matvec_mul(W, std::span<const float>(x));
  }

  static std::vector<float> matvec_mul(const Matrix<float> &W, std::span<const float> x) {
    assert (W.cols() == x.size());
    std::vector<float> result(W.rows(), 0.0f);
    for (size_t i = 0; i < W.rows(); ++i) {
      for (size_t j = 0; j < W.cols(); ++j) {
        result[i] += W(i, j) * x[j];
      }
    }
    return result;
  }

  // Vector-Matrix operations (for gradients: outer product creates matrix)
  static Matrix<float> vec_outer_product(const std::vector<float> &a, const std::vector<float> &b) {
    Matrix<float> result(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
      for (size_t j = 0; j < b.size(); ++j) {
        result(i, j) = a[i] * b[j];
      }
    }
    return result;
  }

  // Matrix-Matrix operations (already in your Matrix class, but adding utilities)
  static Matrix<float> hadamard_matrix(const Matrix<float> &a, const Matrix<float> &b) {
    assert (a.rows() == b.rows() && a.cols() == b.cols());

    Matrix<float> result(a.rows(), a.cols());
    for (size_t i = 0; i < a.rows(); ++i) {
      for (size_t j = 0; j < a.cols(); ++j) {
        result(i, j) = a(i, j) * b(i, j);
      }
    }
    return result;
  }

  // Vector operations (for compatibility with your RNN interface)
  static std::vector<float> matvec_mul(const std::vector<std::vector<float>> &W, const std::vector<float> &x) {
    assert (!W.empty() && W[0].size() == x.size());

    std::vector<float> result(W.size(), 0.0f);
    for (size_t i = 0; i < W.size(); ++i) {
      for (size_t j = 0; j < x.size(); ++j) {
        result[i] += W[i][j] * x[j];
      }
    }
    return result;
  }

  static std::vector<float> add(const std::vector<float> &a, const std::vector<float> &b) {
    assert (a.size() == b.size());

    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      result[i] = a[i] + b[i];
    }
    return result;
  }

  static std::vector<float> subtract(const std::vector<float> &a, const std::vector<float> &b) {
    assert (a.size() == b.size());

    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      result[i] = a[i] - b[i];
    }
    return result;
  }

  static std::vector<float> hadamard(const std::vector<float> &a, const std::vector<float> &b) {
    assert (a.size() == b.size());

    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      result[i] = a[i] * b[i];
    }
    return result;
  }

  static std::vector<std::vector<float>> outer_product(const std::vector<float> &a,
    const std::vector<float> &b) {
    std::vector<std::vector<float>> result(a.size(), std::vector<float>(b.size()));
    for (size_t i = 0; i < a.size(); ++i) {
      for (size_t j = 0; j < b.size(); ++j) {
        result[i][j] = a[i] * b[j];
      }
    }
    return result;
  }

  static std::vector<float> scalar_multiply(const std::vector<float> &vec, float scalar) {
    std::vector<float> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
      result[i] = vec[i] * scalar;
    }
    return result;
  }

  static std::vector<std::vector<float>> to_vector_matrix(const Matrix<float> &mat) {
    std::vector<std::vector<float>> result(mat.rows(), std::vector<float>(mat.cols()));
    for (size_t i = 0; i < mat.rows(); ++i) {
      for (size_t j = 0; j < mat.cols(); ++j) {
        result[i][j] = mat(i, j);
      }
    }
    return result;
  }
};

#if 0
// 4D Tensor wrapper for easier handling
template<typename T>
class Tensor4D {
public:
  std::vector<T> data;
  size_t batch, channels, height, width;

  Tensor4D(size_t b, size_t c, size_t h, size_t w)
    : batch(b), channels(c), height(h), width(w) {
    data.resize(b * c * h * w);
  }

  // Access: tensor(batch, channel, row, col)
  T &operator()(size_t b, size_t c, size_t h, size_t w) {
    assert(b < batch && c < channels && h < height && w < width);
    return data[b * channels * height * width + c * height * width + h * width + w];
  }

  const T &operator()(size_t b, size_t c, size_t h, size_t w) const {
    assert(b < batch && c < channels && h < height && w < width);
    return data[b * channels * height * width + c * height * width + h * width + w];
  }

  // Get flat vector for MLP integration
  std::vector<T> flatten() const {
    return data;
  }

  // Get dimensions
  std::vector<size_t> shape() const {
    return { batch, channels, height, width };
  }
};
#endif

namespace utils {

  inline std::vector<float> convolve1D(
    const std::vector<float> &input,
    const std::vector<float> &kernel,
    int stride = 1,
    int padding = 0) {
    if (input.empty() || kernel.empty()) return {};
    int input_size = static_cast<int>(input.size());
    int kernel_size = static_cast<int>(kernel.size());
    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;
    std::vector<float> output(output_size, 0.0f);

    for (int i = 0; i < output_size; ++i) {
      for (int j = 0; j < kernel_size; ++j) {
        int input_index = i * stride + j - padding;
        if (input_index >= 0 && input_index < input_size) {
          output[i] += input[input_index] * kernel[j];
        }
      }
    }

    return output;
  }


  inline std::vector<float> convolve2D_separable(
    const std::vector<float> &input,
    int width, int height,
    const std::vector<float> &kernel1D,
    int padding = 1) {

    std::vector<float> temp(width * height);

    // Horizontal convolution
    for (int y = 0; y < height; ++y) {
      std::vector<float> row(width);
      for (int x = 0; x < width; ++x) {
        row[x] = input[y * width + x];
      }
      std::vector<float> conv_row = convolve1D(row, kernel1D, 1, padding);
      for (int x = 0; x < width; ++x) {
        temp[y * width + x] = conv_row[x];
      }
    }

    // Vertical convolution
    std::vector<float> result(width * height);
    for (int x = 0; x < width; ++x) {
      std::vector<float> col(height);
      for (int y = 0; y < height; ++y) {
        col[y] = temp[y * width + x];
      }
      std::vector<float> conv_col = convolve1D(col, kernel1D, 1, padding);
      for (int y = 0; y < height; ++y) {
        result[y * width + x] = conv_col[y];
      }
    }

    return result;
  }


  inline std::vector<float> convolve2D(
    const Matrix<float> &input,
    const Matrix<float> &kernel,
    int stride = 1,
    int padding = 0) {
    if (input.empty() || kernel.empty()) return {};
    const auto width = input.cols();
    const auto height = input.rows();
    const auto kernelWidth = kernel.cols();
    const auto kernelHeight = kernel.rows();
    const auto outputWidth = (width - kernelWidth + 2 * padding) / stride + 1;
    const auto outputHeight = (height - kernelHeight + 2 * padding) / stride + 1;
    std::vector<float> output(outputWidth * outputHeight, 0.0f);
    for (int y = 0; y < outputHeight; ++y) {
      for (int x = 0; x < outputWidth; ++x) {
        for (int ky = 0; ky < kernelHeight; ++ky) {
          for (int kx = 0; kx < kernelWidth; ++kx) {
            int input_x = x * stride + kx - padding;
            int input_y = y * stride + ky - padding;
            if (input_x >= 0 && input_x < width && input_y >= 0 && input_y < height) {
              output[y * outputWidth + x] += input/*[input_y * width + input_x]*/(input_y, input_x) * kernel/*[ky * kernelWidth + kx]*/(ky, kx);
            }
          }
        }
      }
    }
    return output;
  }

} // namespace utils

