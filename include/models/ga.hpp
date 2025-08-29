#pragma once
#include <vector>
#include <functional>
#include <random>


namespace ga {

  struct Chromosome {
    std::vector<float> genes;
    float fitness = 0.0f;
  };

  class GeneticAlgorithm {
  public:
    GeneticAlgorithm(
      int population_size,
      int gene_length,
      int max_generations,
      float crossover_rate,
      float mutation_rate,
      std::function<float(const std::vector<float> &)> fitness_fn,
      float init_min = 0.0f,
      float init_max = 1.0f
    );

    void run();  // evolves the population

    const Chromosome &bestSolution() const { return best_; }
    const std::vector<float> &history() const { return history_; }

  private:
    int populationSize_;
    int geneLength_;
    int maxGenerations_;
    float crossoverRate_;
    float mutationRate_;
    float initMin_;
    float initMax_;
    std::vector<float> history_;

    std::function<float(const std::vector<float> &)> fitness_fn_;

    std::vector<Chromosome> population_;
    Chromosome best_;

    std::mt19937 rng_;

    // Core steps
    void initializePopulation();
    Chromosome selectParent();
    Chromosome crossover(const Chromosome &p1, const Chromosome &p2);
    void mutate(Chromosome &child);
    void evaluatePopulation();
  };
} // namespace ga
