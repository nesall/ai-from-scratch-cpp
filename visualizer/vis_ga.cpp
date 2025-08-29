#include "plotter.hpp"
#include "models/ga.hpp"

std::pair<float, float> plotGA(const ga::GeneticAlgorithm &ga, SimplePlotter &plotter) {
  const auto &hist = ga.history();
  float minVal = *std::min_element(hist.begin(), hist.end());
  float maxVal = *std::max_element(hist.begin(), hist.end());
  plotter.setDataBounds(0.f, static_cast<float>(hist.size()), minVal, maxVal);
  for (size_t i = 1; i < hist.size(); ++i) {
    plotter.addLine(
      sf::Vector2f(plotter.toScreenX(static_cast<float>(i - 1)), plotter.toScreenY(hist[i - 1])),
      sf::Vector2f(plotter.toScreenX(static_cast<float>(i)), plotter.toScreenY(hist[i]))
    );
  }
  return { minVal, maxVal };
}



void vis_ga(SimplePlotter &plotter) {
  plotter.setWindowSize(800, 600);
  plotter.clearShapes();
  plotter.setDrawAxes(true);

  auto fitness = [](const std::vector<float> &genes) {
    float x = genes[0] * 10.0f;  // search in [0,10]
    return std::sin(x) * x;
    };

  ga::GeneticAlgorithm ga(50, 1, 100, 0.7f, 0.1f, fitness);
  ga.run();

  std::cout << "[GA] Best solution: x=" << ga.bestSolution().genes[0] * 10.0f
    << ", fitness=" << ga.bestSolution().fitness << "\n";

  const auto [minVal, maxVal] = plotGA(ga, plotter);

  plotter.addText(sf::Vector2f(plotter.toScreenX(50), plotter.toScreenY(maxVal)), "Optimizing f(x) = sin(x) * x", sf::Color::Black);

}