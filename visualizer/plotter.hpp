#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <algorithm>


struct DrawContext {
  sf::RenderWindow *window = nullptr;
  virtual float toScreenX(float) = 0;
  virtual float toScreenY(float) = 0;
};

struct Shape {
  virtual ~Shape() = default;
  virtual void draw(DrawContext &ctx) const = 0;
  virtual sf::FloatRect boundingBox() const = 0;
  virtual sf::Vector2f center() const = 0;
};

struct Circle : public Shape {
  sf::CircleShape circle;
  Circle(float radius, const sf::Vector2f &position, const sf::Color &color) {
    circle.setRadius(radius);
    circle.setPosition(position);
    circle.setFillColor(color);
  }
  void draw(DrawContext &ctx) const override {
    sf::CircleShape c{ circle };
    ctx.window->draw(c);
  }
  sf::FloatRect boundingBox() const override {
    return circle.getGlobalBounds();
  }
  sf::Vector2f center() const override {
    return circle.getPosition() + sf::Vector2f(circle.getRadius(), circle.getRadius());
  }
};

struct Line : public Shape {
  sf::Vertex line[2];
  Line(const sf::Vector2f &start, const sf::Vector2f &end, const sf::Color &color) {
    line[0] = sf::Vertex(start, color);
    line[1] = sf::Vertex(end, color);
  }
  void draw(DrawContext &ctx) const override {
    ctx.window->draw(line, 2, sf::Lines);
  }
  sf::FloatRect boundingBox() const override {
    return sf::FloatRect(std::min(line[0].position.x, line[1].position.x),
      std::min(line[0].position.y, line[1].position.y),
      std::abs(line[1].position.x - line[0].position.x),
      std::abs(line[1].position.y - line[0].position.y));
  }
  sf::Vector2f center() const override {
    return (line[0].position + line[1].position) / 2.f;
  }
};



class SimplePlotter : public DrawContext {
  sf::RenderWindow window;
  sf::Font font;
  float margin = 50.f;
  sf::Color backgroundColor = sf::Color::White;
  sf::Color axisColor = sf::Color::Black;
  sf::Color dataColor = sf::Color::Blue;
  sf::Color fitColor = sf::Color::Red;

  std::vector<std::unique_ptr<Shape>> shapes;

  float dataBoundsMinX = 0.f;
  float dataBoundsMaxX = 10.f;
  float dataBoundsMinY = 0.f;
  float dataBoundsMaxY = 10.f;

public:
  SimplePlotter(int width = 800, int height = 600)
    : window(sf::VideoMode(width, height), "Fitting Algorithm Visualizer") {
    // Load default font (you might want to load a specific font file)
    if (!font.loadFromFile("arial.ttf")) {
      // Handle font loading error - use default font
    }
    DrawContext::window = &window;
  }

  void addShape(std::unique_ptr<Shape> shape) {
    shapes.push_back(std::move(shape));
  }

  void addCircle(float radius, const sf::Vector2f &position, const sf::Color &color = sf::Color::Red) {
    auto pos{ position };
    pos.x -= radius;
    pos.y -= radius; // Center the circle at the position
    addShape(std::make_unique<Circle>(radius, pos, color));
  }

  void addLine(const sf::Vector2f &start, const sf::Vector2f &end, const sf::Color &color = sf::Color::Blue) {
    addShape(std::make_unique<Line>(start, end, color));
  }

  void clearShapes() {
    shapes.clear();
  }

  Shape *refShape(size_t index) {
    if (index < shapes.size()) {
      return shapes[index].get();
    }
    return nullptr; // Or throw an exception
  }

  size_t nofShapes() const {
    return shapes.size();
  }

  void drawNextFrame() {
    float plotWidth = window.getSize().x - 2 * margin;
    float plotHeight = window.getSize().y - 2 * margin;

    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed)
        window.close();
    }

    window.clear(backgroundColor);

    drawAxes(plotWidth, plotHeight);

    for (const auto &shape : shapes) {
      shape->draw(*this);
    }

    window.display();
  }

  void start() {
    while (window.isOpen()) {
      drawNextFrame();
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

  float toScreen(float x, float minX, float maxX, float plotWidth) {
    return margin + ((x - minX) / (maxX - minX)) * plotWidth;
  }

public:
  void setDataBounds(float minx, float maxx, float miny, float maxy) {
    dataBoundsMaxX = std::max(minx, maxx);
    dataBoundsMinX = std::min(minx, maxx);
    dataBoundsMaxY = std::max(miny, maxy);
    dataBoundsMinY = std::min(miny, maxy);
  }

  float toScreenX(float x) override {
    return toScreen(x, dataBoundsMinX, dataBoundsMaxX, window.getSize().x - 2 * margin);
  }

  float toScreenY(float y) override {
    return toScreen(y, dataBoundsMinY, dataBoundsMaxY, window.getSize().y - 2 * margin);
  }
};