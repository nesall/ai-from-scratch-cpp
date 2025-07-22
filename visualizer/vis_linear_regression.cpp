#include <thread>
#include <chrono>
#include <SFML/Graphics.hpp>

#include "plotter.hpp"
#include "../include/models/linear_regression.hpp"
#include "../include/datasynth.hpp"


void vis_linear_regression(SimplePlotter &plotter) {
  const int num_samples = 100;
  auto pts = generate_synthetic_data<float>(num_samples, 2.0f, 1.0f, 0.f, 10.f, 0.5f);

  auto mx = std::minmax_element(pts.cbegin(), pts.cend(), [](auto a, auto b) { return a.x < b.x; });
  auto my = std::minmax_element(pts.cbegin(), pts.cend(), [](auto a, auto b) { return a.y < b.y; });

  plotter.setDataBounds(mx.first->x, mx.second->x, my.first->y, my.second->y);

  for (const auto &pt : pts) {
    auto xy = sf::Vector2f(plotter.toScreenX(pt.x), plotter.toScreenY(pt.y));
    plotter.addCircle(3.f, xy, sf::Color::Blue);
  }

#if 1
  models::LinearRegression linreg(0.01f, 300);
  linreg.setStepCallback([&]()
    {

      float y0 = linreg.predict(1.f);
      float y1 = linreg.predict(9.f);

      if (0 < plotter.nofShapes()) {
        auto start = sf::Vector2f(plotter.toScreenX(1.f), plotter.toScreenY(y0));
        auto end = sf::Vector2f(plotter.toScreenX(9.f), plotter.toScreenY(y1));
        auto line = dynamic_cast<Line *>(plotter.refShape(plotter.nofShapes() - 1));
        if (line) {
          line->line[0] = sf::Vertex(start, sf::Color::Red);
          line->line[1] = sf::Vertex(end, sf::Color::Red);
          line->boundingBox(); // Force recalculation of bounding box
        } else {
          plotter.addLine(start, end, sf::Color::Red);
        }
      }

      plotter.drawNextFrame();

      std::this_thread::sleep_for(std::chrono::milliseconds(33)); // Control the update rate

      return true; // Continue the fitting process
    }
  );
  linreg.fit(pts);
#endif

}