#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <cassert>


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

  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t size() const { return data_.size(); }
  bool empty() const { return data_.empty(); }

  //void print() const;

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

