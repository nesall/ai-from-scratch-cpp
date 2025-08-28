#include <vector>
#include <algorithm>
#include <cassert>

#include "matrix.hpp"
#include "models/cnn.hpp"
#include "plotter.hpp"


void vis_convolution(SimplePlotter &plotter) {
  plotter.clearShapes();
  plotter.setDrawAxes(false);
  std::string path = "../../data/mario-transparent.png";
  //plotter.setDrawImage(path);

  sf::Image image;
  if (image.loadFromFile(path)) {
    const auto width = image.getSize().x;
    const auto height = image.getSize().y;
    const sf::Uint8 *data = image.getPixelsPtr();
    assert(data);
    // Each pixel has 4 components (RGBA)
    const std::size_t dataLength = width * height * 4;


    // prepare for convolution using utils::convolve to achieve a gaussian blur effect

    std::vector<float> input(dataLength / 4);
    for (std::size_t i = 0; i < dataLength; i += 4) {
      // Convert RGBA to grayscale
      input[i / 4] = 0.299f * data[i] + 0.587f * data[i + 1] + 0.114f * data[i + 2];
    }

    assert(input.size() == width * height);

    // Define a Gaussian kernel
    std::vector<float> kernel = {
      1.0f / 16, 2.0f / 16, 4.0f / 16, 2.f / 16, 1.0f / 16,
      //2.0f / 16, 4.0f / 16, 2.0f / 16,
      //1.0f / 16, 2.0f / 16, 1.0f / 16
    };
    std::vector<float> output = utils::convolve2D_separable(input, width, height, kernel, 3);
    assert(output.size() == width * height);
    // Create a new image to store the convolved result
    sf::Image convolvedImage;
    convolvedImage.create(width, height, sf::Color::Black);
    for (std::size_t i = 0; i < output.size(); ++i) {
      // Convert grayscale back to RGBA
      sf::Uint8 gray = static_cast<sf::Uint8>(std::clamp(output[i], 0.0f, 255.0f));
      convolvedImage.setPixel(i % width, i / width, sf::Color(gray, gray, gray, 255));
    }

    convolvedImage.saveToFile("mario-blurred.png");
    plotter.setDrawImage("mario-blurred.png");
  }
}
