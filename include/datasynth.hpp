#pragma once

#include "common.hpp"
#include <vector>
#include <cmath>
#include <random>


/**
 * Generates synthetic linear regression data: y = w*x + b + noise
 * @param pts Output vector for generated points
 * @param num_points Number of points to generate
 * @param true_w True slope coefficient
 * @param true_b True intercept
 * @param rangeMin Minimum x value
 * @param rangeMax Maximum x value
 * @param noise_std Standard deviation of Gaussian noise
 * @param seed Random seed for reproducible results
 */
template<typename T>
void generate_synthetic_data(std::vector<commons::Point<T>>& pts, 
                           int num_points, 
                           T true_w, 
                           T true_b, 
                           T rangeMin = static_cast<T>(0), 
                           T rangeMax = static_cast<T>(10), 
                           T noise_std = static_cast<T>(1), 
                           unsigned int seed = 42) {
    
    // Parameter validation
    if (num_points <= 0) return;
    if (rangeMin >= rangeMax) {
        std::swap(rangeMin, rangeMax);
    }
    if (noise_std < static_cast<T>(0)) {
        noise_std = static_cast<T>(0);
    }
    
    // Pre-allocate memory
    pts.reserve(pts.size() + num_points);
    
    // Setup random number generators
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<T> dist_x(rangeMin, rangeMax);
    std::normal_distribution<T> dist_noise(static_cast<T>(0), noise_std);
    
    // Generate points
    for (int i = 0; i < num_points; ++i) {
        T xi = dist_x(eng);
        T yi = true_w * xi + true_b + dist_noise(eng);
        pts.emplace_back(xi, yi);
    }
}

template<typename T>
std::vector<commons::Point<T>> generate_synthetic_data(int num_points,
  T true_w,
  T true_b,
  T rangeMin = static_cast<T>(0),
  T rangeMax = static_cast<T>(10),
  T noise_std = static_cast<T>(1),
  unsigned int seed = 42) {

  std::vector<commons::Point<T>> pts;
  generate_synthetic_data(pts, num_points, true_w, true_b, rangeMin, rangeMax, noise_std, seed);
  return pts;
}