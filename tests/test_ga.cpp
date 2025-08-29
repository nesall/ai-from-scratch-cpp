#include <iostream>
#include <cmath>
#include <vector>
#include "models/ga.hpp"

float fitness_function(const std::vector<float> &genes) {
  float x = genes[0];
  // Penalize values outside the [0, 10] range
  if (x < 0 || x > 10) return -1e6f;
  return std::sin(x) * x;
}

void test_ga() {
  using namespace ga;

  GeneticAlgorithm ga(
    /* population_size */ 50,
    /* gene_length */ 1,
    /* max_generations */ 100,
    /* crossover_rate */ 0.7f,
    /* mutation_rate */ 0.1f,
    fitness_function, 0, 10.f
  );

  ga.run();

  const Chromosome &best = ga.bestSolution();

  std::cout << "[GeneticAlgorithm] Best solution found:\n";
  std::cout << "  x = " << best.genes[0] << ", fitness = " << best.fitness << "\n";

  // Expected: x near ~7.85 (since sin(x)*x peaks near 7.85 with value ~7.85)
}
