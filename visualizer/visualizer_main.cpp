#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

class SimplePlotter {
private:
  sf::RenderWindow window;
  sf::Font font;
  float margin = 50.f;
  sf::Color backgroundColor = sf::Color::White;
  sf::Color axisColor = sf::Color::Black;
  sf::Color dataColor = sf::Color::Blue;
  sf::Color fitColor = sf::Color::Red;

public:
  SimplePlotter(int width = 800, int height = 600)
    : window(sf::VideoMode(width, height), "Fitting Algorithm Visualizer") {
    // Load default font (you might want to load a specific font file)
    if (!font.loadFromFile("arial.ttf")) {
      // Handle font loading error - use default font
    }
  }

  void plotData(const std::vector<sf::Vector2f> &dataPoints,
    const std::vector<sf::Vector2f> &fittedCurve = {}) {

    if (dataPoints.empty()) return;

    // Find data bounds
    auto [minX, maxX] = std::minmax_element(dataPoints.begin(), dataPoints.end(),
      [](const sf::Vector2f &a, const sf::Vector2f &b) { return a.x < b.x; });
    auto [minY, maxY] = std::minmax_element(dataPoints.begin(), dataPoints.end(),
      [](const sf::Vector2f &a, const sf::Vector2f &b) { return a.y < b.y; });

    float dataWidth = maxX->x - minX->x;
    float dataHeight = maxY->y - minY->y;

    // Add some padding
    float xPadding = dataWidth * 0.1f;
    float yPadding = dataHeight * 0.1f;

    float plotWidth = window.getSize().x - 2 * margin;
    float plotHeight = window.getSize().y - 2 * margin;

    while (window.isOpen()) {
      sf::Event event;
      while (window.pollEvent(event)) {
        if (event.type == sf::Event::Closed)
          window.close();
      }

      window.clear(backgroundColor);

      // Draw axes
      drawAxes(plotWidth, plotHeight);

      // Draw data points
      drawPoints(dataPoints, *minX, *maxX, *minY, *maxY,
        plotWidth, plotHeight, xPadding, yPadding);

      // Draw fitted curve if provided
      if (!fittedCurve.empty()) {
        drawCurve(fittedCurve, *minX, *maxX, *minY, *maxY,
          plotWidth, plotHeight, xPadding, yPadding);
      }

      window.display();
    }
  }

private:
  void drawAxes(float plotWidth, float plotHeight) {
    // X-axis
    sf::RectangleShape xAxis(sf::Vector2f(plotWidth, 2.f));
    xAxis.setPosition(margin, margin + plotHeight);
    xAxis.setFillColor(axisColor);
    window.draw(xAxis);

    // Y-axis
    sf::RectangleShape yAxis(sf::Vector2f(2.f, plotHeight));
    yAxis.setPosition(margin, margin);
    yAxis.setFillColor(axisColor);
    window.draw(yAxis);
  }

  void drawPoints(const std::vector<sf::Vector2f> &points,
    const sf::Vector2f &minBounds, const sf::Vector2f &maxBounds,
    const sf::Vector2f &minY, const sf::Vector2f &maxY,
    float plotWidth, float plotHeight,
    float xPadding, float yPadding) {

    for (const auto &point : points) {
      sf::CircleShape circle(3.f);

      // Transform to screen coordinates
      float screenX = margin + ((point.x - minBounds.x + xPadding) /
        (maxBounds.x - minBounds.x + 2 * xPadding)) * plotWidth;
      float screenY = margin + plotHeight - ((point.y - minY.y + yPadding) /
        (maxY.y - minY.y + 2 * yPadding)) * plotHeight;

      circle.setPosition(screenX - 3.f, screenY - 3.f);
      circle.setFillColor(dataColor);
      window.draw(circle);
    }
  }

  void drawCurve(const std::vector<sf::Vector2f> &curve,
    const sf::Vector2f &minBounds, const sf::Vector2f &maxBounds,
    const sf::Vector2f &minY, const sf::Vector2f &maxY,
    float plotWidth, float plotHeight,
    float xPadding, float yPadding) {

    if (curve.size() < 2) return;

    for (size_t i = 0; i < curve.size() - 1; ++i) {
      sf::Vertex line[] = {
          sf::Vertex(transformToScreen(curve[i], minBounds, maxBounds, minY, maxY,
                                     plotWidth, plotHeight, xPadding, yPadding), fitColor),
          sf::Vertex(transformToScreen(curve[i + 1], minBounds, maxBounds, minY, maxY,
                                     plotWidth, plotHeight, xPadding, yPadding), fitColor)
      };
      window.draw(line, 2, sf::Lines);
    }
  }

  sf::Vector2f transformToScreen(const sf::Vector2f &point,
    const sf::Vector2f &minBounds, const sf::Vector2f &maxBounds,
    const sf::Vector2f &minY, const sf::Vector2f &maxY,
    float plotWidth, float plotHeight,
    float xPadding, float yPadding) {
    float screenX = margin + ((point.x - minBounds.x + xPadding) /
      (maxBounds.x - minBounds.x + 2 * xPadding)) * plotWidth;
    float screenY = margin + plotHeight - ((point.y - minY.y + yPadding) /
      (maxY.y - minY.y + 2 * yPadding)) * plotHeight;
    return sf::Vector2f(screenX, screenY);
  }
};

// Example usage
int main() {
  SimplePlotter plotter(800, 600);

  // Generate some sample data
  std::vector<sf::Vector2f> dataPoints;
  std::vector<sf::Vector2f> fittedCurve;

  // Sample noisy data
  for (float x = 0; x <= 10; x += 0.5f) {
    float noise = (rand() % 100 - 50) / 100.0f;
    dataPoints.push_back({ x, 2 * x + 1 + noise });
  }

  // Sample fitted line
  for (float x = 0; x <= 10; x += 0.1f) {
    fittedCurve.push_back({ x, 2 * x + 1 });
  }

  plotter.plotData(dataPoints, fittedCurve);

  return 0;
}
