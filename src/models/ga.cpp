#include "models/ga.hpp"


ga::GeneticAlgorithm::GeneticAlgorithm(
  int population_size, int gene_length, int max_generations, float crossover_rate, float mutation_rate, 
  std::function<float(const std::vector<float> &)> fitness_fn, float init_min , float init_max)
  : populationSize_(population_size), geneLength_(gene_length), maxGenerations_(max_generations), crossoverRate_(crossover_rate), mutationRate_(mutation_rate), fitness_fn_(fitness_fn), rng_(std::random_device{}()), initMin_(init_min), initMax_(init_max)  
{
}

void ga::GeneticAlgorithm::initializePopulation()
{
  std::uniform_real_distribution<float> dist(initMin_, initMax_);
  population_.resize(populationSize_);
  for (auto &chrom : population_) {
    chrom.genes.resize(geneLength_);
    for (auto &gene : chrom.genes) {
      gene = dist(rng_); // Random gene in [0,1]
    }
    chrom.fitness = 0.0f;
  }
}

ga::Chromosome ga::GeneticAlgorithm::selectParent()
{
  Chromosome chr;
  // Tournament selection
  int tournament_size = 3;
  std::uniform_int_distribution<int> dist(0, populationSize_ - 1);
  chr = population_[dist(rng_)];
  for (int i = 1; i < tournament_size; ++i) {
    Chromosome contender = population_[dist(rng_)];
    if (contender.fitness > chr.fitness) {
      chr = contender;
    }
  }
  return chr;
}

ga::Chromosome ga::GeneticAlgorithm::crossover(const Chromosome &p1, const Chromosome &p2)
{
  Chromosome chr;
  chr.genes.resize(geneLength_);
  std::uniform_int_distribution<int> dist(0, geneLength_ - 1);
  int crossover_point = dist(rng_);
  for (int i = 0; i < geneLength_; ++i) {
    if (i < crossover_point) {
      chr.genes[i] = p1.genes[i];
    } else {
      chr.genes[i] = p2.genes[i];
    }
  }
  return chr;
}

void ga::GeneticAlgorithm::mutate(Chromosome &child)
{
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto &gene : child.genes) {
    if (dist(rng_) < mutationRate_) {
      gene = dist(rng_); // Random mutation
    }
  }
}

void ga::GeneticAlgorithm::evaluatePopulation()
{
  for (auto &chrom : population_) {
    chrom.fitness = fitness_fn_(chrom.genes);
    if (chrom.fitness > best_.fitness) {
      best_ = chrom;
    }
  }
}

void ga::GeneticAlgorithm::run()
{
  initializePopulation();
  for (int gen = 0; gen < maxGenerations_; ++gen) {
    evaluatePopulation();
    history_.push_back(best_.fitness);
    // Create new population
    std::vector<Chromosome> new_population;
    while (new_population.size() < populationSize_) {
      Chromosome parent1 = selectParent();
      Chromosome parent2 = selectParent();
      Chromosome child;
      if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_) < crossoverRate_) {
        child = crossover(parent1, parent2);
      } else {
        child = parent1; // No crossover, just copy
      }
      mutate(child);
      new_population.push_back(child);
    }
    population_ = std::move(new_population);
  }
  evaluatePopulation(); // Final evaluation
  history_.push_back(best_.fitness);
}