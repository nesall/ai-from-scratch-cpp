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
