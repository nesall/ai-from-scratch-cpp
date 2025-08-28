#include <SFML/Graphics.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
#include <thread>
#include <filesystem>
#include <functional>
#include <cmath>


struct DrawContext {
  sf::RenderWindow *window_ = nullptr;
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
    ctx.window_->draw(c);
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
    ctx.window_->draw(line, 2, sf::Lines);
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

struct Text : public Shape {
  sf::Text text;
  Text(const sf::Vector2f &position, const sf::Color &color, const sf::Font &font, const std::string &str, unsigned int charSize = 12) {
    text.setFont(font);
    text.setString(str);
    text.setCharacterSize(charSize);
    text.setFillColor(color);
    text.setPosition(position);
    text.setString(str);
  }
  void draw(DrawContext &ctx) const override {
    ctx.window_->draw(text);
  }
  sf::FloatRect boundingBox() const override {
    return text.getGlobalBounds();
  }
  sf::Vector2f center() const override {
    auto bounds = text.getGlobalBounds();
    return sf::Vector2f(bounds.left + bounds.width / 2.f, bounds.top + bounds.height / 2.f);
  }
};



class SimplePlotter : public DrawContext {
  sf::RenderWindow window_;
  sf::Font font;
  const float margin_ = 50.f;
  sf::Color backgroundColor = sf::Color::White;
  sf::Color axisColor = sf::Color::Black;
  sf::Color dataColor = sf::Color::Blue;
  sf::Color fitColor = sf::Color::Red;

  std::vector<std::unique_ptr<Shape>> shapes_;

  float dataBoundsMinX = 0.f;
  float dataBoundsMaxX = 10.f;
  float dataBoundsMinY = 0.f;
  float dataBoundsMaxY = 10.f;

  bool drawAxes_ = true;
  sf::Texture bgTexture_;


  struct TabBtn {
    std::string name;
    std::function<void()> func;
  };
  std::vector<TabBtn> tabs_;

public:
  SimplePlotter(unsigned int width = 800, unsigned int height = 600)
    : window_(sf::VideoMode(width, height), "Fitting Algorithm Visualizer") {
    // Load default font (you might want to load a specific font file)
    if (!font.loadFromFile("../../data/arial.ttf")) {
      // Handle font loading error - use default font
    }
    DrawContext::window_ = &window_;
  }

  void addShape(std::unique_ptr<Shape> shape) {
    shapes_.push_back(std::move(shape));
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

  void addText(const sf::Vector2f &position, const std::string &str, const sf::Color &color = sf::Color::Black) {
    addShape(std::make_unique<Text>(position, color, font, str));
  }

  void clearShapes() {
    shapes_.clear();
    bgTexture_ = {};
  }

  Shape *refShape(size_t index) {
    if (index < shapes_.size()) {
      return shapes_[index].get();
    }
    return nullptr; // Or throw an exception
  }

  size_t nofShapes() const {
    return shapes_.size();
  }

  void drawNextFrame() {
    float plotWidth = window_.getSize().x - 2 * margin_;
    float plotHeight = window_.getSize().y - 2 * margin_;

    sf::Event event;
    while (window_.pollEvent(event)) {
      if (event.type == sf::Event::Closed)
        window_.close();
      else if (event.type == sf::Event::MouseMoved) {
        onMouseMove(event.mouseMove.x, event.mouseMove.y);
      } else if (event.type == sf::Event::MouseButtonPressed) {
        onMouseButtonPress(event.mouseButton.button, event.mouseButton.x, event.mouseButton.y);
      }
    }

    window_.clear(backgroundColor);

    sf::Sprite sprite(bgTexture_);
    if (sprite.getTexture()) {
      sprite.setPosition(margin_, margin_);
      window_.draw(sprite);
    }

    if (drawAxes_)
      drawAxes(plotWidth, plotHeight);

    for (const auto &shape : shapes_) {
      shape->draw(*this);
    }

    drawTabs();

    window_.display();
  }

  void setDrawAxes(bool f) {
    drawAxes_ = f;
  }

  void setDrawImage(const std::string &path) {
    if (path.empty()) {
      std::cout << "No image path provided." << std::endl;
      return; // No image to draw
    }
    if (!std::filesystem::exists(path)) {
      std::cout << "File path " << std::filesystem::absolute(path) << " does not exist." << std::endl;
      return;
    }

    if (bgTexture_.loadFromFile(path)) {
      const auto w = bgTexture_.getSize().x + 2 * margin_;
      const auto h = bgTexture_.getSize().y + 2 * margin_;
      window_.setSize(sf::Vector2u(w, h));
      // After resizing, set the view to match the drawable area:
      sf::View view(sf::FloatRect(0, 0, w, h));
      window_.setView(view);
    } else {
      std::cout << "Unable to load texture." << std::endl;
      // Handle error loading image
    }
  }

  void setWindowSize(unsigned w, unsigned h) {
    window_.setSize(sf::Vector2u{ w, h });
    sf::View view(sf::FloatRect(0, 0, w, h));
    window_.setView(view);
  }

  void start() {
    while (window_.isOpen()) {
      drawNextFrame();
      std::this_thread::sleep_for(std::chrono::milliseconds(16)); // Control the frame rate
    }
  }

  // simple GUI
  void addTabBtn(const std::string &s, std::function<void()> f) {
    tabs_.emplace_back(s, f);
  }

private:
  void drawAxes(float plotWidth, float plotHeight) {
    // X-axis
    sf::RectangleShape xAxis(sf::Vector2f(plotWidth, 2.f));
    xAxis.setPosition(margin_, margin_ + plotHeight);
    xAxis.setFillColor(axisColor);
    window_.draw(xAxis);

    // Y-axis
    sf::RectangleShape yAxis(sf::Vector2f(2.f, plotHeight));
    yAxis.setPosition(margin_, margin_);
    yAxis.setFillColor(axisColor);
    window_.draw(yAxis);
  }

  float toScreen(float x, float minX, float maxX, float plotWidth) {
    return margin_ + ((x - minX) / (maxX - minX)) * plotWidth;
  }

  void drawTabs() {
    const float padding = 5;
    float tabbarWidth = window_.getSize().x - 2 * padding;
    size_t nTabs = tabs_.size();
    float totalPadding = (0 < nTabs) ? (nTabs - 1) * padding : 0;
    float tabWidth = (tabbarWidth - totalPadding) / nTabs;
    float tabHeight = margin_ - 2 * padding;
    float x = padding;
    float y = padding;
    for (const auto &tab : tabs_) {
      sf::RectangleShape tabRect(sf::Vector2f(tabWidth - padding, tabHeight));
      tabRect.setPosition(x, y);
      tabRect.setFillColor(sf::Color(200, 200, 200));
      window_.draw(tabRect);
      sf::Text tabText;
      tabText.setFont(font);
      tabText.setString(tab.name);
      tabText.setCharacterSize(14);
      tabText.setFillColor(sf::Color::Black);
      sf::FloatRect textBounds = tabText.getLocalBounds();
      tabText.setPosition(x + (tabWidth - padding - textBounds.width) / 2.f, y + (tabHeight - textBounds.height) / 2.f - 5);
      window_.draw(tabText);
      // Handle mouse click
      if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
        sf::Vector2i mousePos = sf::Mouse::getPosition(window_);
        if (mousePos.x >= x && mousePos.x <= x + tabWidth - padding &&
          mousePos.y >= y && mousePos.y <= y + tabHeight) {
          tab.func(); // Call the associated function
          // Simple debounce
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
      }
      x += tabWidth;
    }
  }

public:
  void setDataBounds(float minx, float maxx, float miny, float maxy) {
    dataBoundsMaxX = std::max(minx, maxx);
    dataBoundsMinX = std::min(minx, maxx);
    dataBoundsMaxY = std::max(miny, maxy);
    dataBoundsMinY = std::min(miny, maxy);
  }

  float toScreenX(float x) override {
    return toScreen(x, dataBoundsMinX, dataBoundsMaxX, window_.getSize().x - 2 * margin_);
  }

  float toScreenY(float y) override {
    return window_.getSize().y - toScreen(y, dataBoundsMinY, dataBoundsMaxY, window_.getSize().y - 2 * margin_);
  }

  void onMouseMove(int x, int y) {
    // Handle mouse move (e.g., highlight tab, show tooltip, etc.)
    // For simplicity, we won't implement hover effects here

  }

  void onMouseButtonPress(sf::Mouse::Button button, int x, int y) {
    // Handle mouse button press (e.g., select tab, interact with plot, etc.)
    if (button == sf::Mouse::Left) {
      // Check if a tab was clicked
      const float padding = 5;
      float tabbarWidth = window_.getSize().x - 2 * padding;
      size_t nTabs = tabs_.size();
      float totalPadding = (0 < nTabs) ? (nTabs - 1) * padding : 0;
      float tabWidth = (tabbarWidth - totalPadding) / nTabs;
      float tabHeight = margin_ - 2 * padding;
      float tabX = padding;
      float tabY = padding;
      for (const auto &tab : tabs_) {
        if (x >= tabX && x <= tabX + tabWidth - padding &&
          y >= tabY && y <= tabY + tabHeight) {
          tab.func(); // Call the associated function
          // Simple debounce
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
          break;
        }
        tabX += tabWidth;
      }
    }
  }
};