#include "plotter.hpp"
#include "models/kmeans.hpp"
#include <random>

namespace {
  const std::vector<sf::Color> _colors = { sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Magenta };
}

void stepFunc(SimplePlotter &plotter, const Matrix<float> &data, const models::KMeans &km) {
  plotter.clearShapes();
  auto labels = km.predict(data);
  for (size_t i = 0; i < data.rows(); ++i) {
    plotter.addCircle(3.f,
      { plotter.toScreenX(data[i][0]), plotter.toScreenY(data[i][1]) },
      _colors[labels[i] % _colors.size()]
    );
  }
  auto centers = km.centroids();
  for (auto c: centers) {
    plotter.addCircle(6.f, { plotter.toScreenX(c[0]), plotter.toScreenY(c[1]) }, sf::Color::Black);
  }
  plotter.drawNextFrame();
  sf::sleep(sf::milliseconds(100));
}


void vis_kmeans(SimplePlotter &plotter) {
  std::cout << "vis_kmeans\n";
  plotter.setWindowSize(g_W, g_H);
  plotter.clearShapes();
  plotter.setDrawAxes(false);
  plotter.setDataBounds(0.f, 10.f, 0.f, 10.f);

  const int k = 4; // number of clusters
  const int points = 300;
  const int steps = 20;

  // Generate random 2D points
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> distX(0.f, 10.f);
  std::uniform_real_distribution<float> distY(0.f, 10.f);

  Matrix<float> data{points, 2};
  for (auto p : data) {
    p[0] = distX(rng);
    p[1] = distY(rng);
  }

  models::KMeans km(k, steps, true);
  km.setStepCallback([&plotter, data, &km] { stepFunc(plotter, data, km); return true; });
  km.fit(data);
}
