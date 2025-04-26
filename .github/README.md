# Image Stitching for Panorama Generation

[![Python Tests](https://github.com/YourUsername/image-stitching-panorama/actions/workflows/python-tests.yml/badge.svg)](https://github.com/YourUsername/image-stitching-panorama/actions/workflows/python-tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

An advanced image stitching system that creates panoramas from multiple images using computer vision techniques.

## 🔍 Features

- **🔎 Intelligent Image Analysis**: Automatically analyzes and optimizes image parameters
- **🌟 Advanced Feature Detection**: Scale-Invariant Feature Transform (SIFT) with adaptive parameters
- **🔗 Robust Feature Matching**: FLANN-based matching with ratio test for accuracy
- **📊 Match Quality Metrics**: Displays percentage of matched features for quality assessment
- **🎨 Multi-method Blending**: Choose between average and multi-band blending techniques
- **🖥️ Interactive UI**: User-friendly Gradio interface with real-time processing
- **👁️ Visualization Tools**: Displays feature matches for better understanding and debugging
- **⚙️ Parameter Customization**: Fine-tune brightness, contrast, and blur for optimal results
- **💻 Command-line Flexibility**: Run with custom ports, hosts, and sharing options

## 🖼️ Screenshots

![Panorama Example](https://i.imgur.com/example.png)

## 🛠️ Installation

### Prerequisites

- Python 3.7 or later
- Pip package manager

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/YourUsername/image-stitching-panorama.git
cd image-stitching-panorama

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 📖 Documentation

Full documentation is available in the main [README.md](../README.md) file.

## 📊 Example Usage

### Web Interface

```bash
python app.py --share
```

### Command Line Demo

```bash
python demo.py --dir ./path/to/images --output panorama.jpg --blend multiband --preprocess
```

### Programmatic Usage

```python
from app import stitch_images

# Load images using OpenCV
img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")

# Stitch images
result, matches_vis, message = stitch_images(img1, img2, blend_method="multiband")

# Check result
if result is not None:
    print(f"Success: {message}")
    cv2.imwrite("panorama.jpg", result)
else:
    print(f"Failed: {message}")
```

## 🧪 Testing

```bash
python test_panorama.py
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 👏 Acknowledgments

- OpenCV library for computer vision algorithms
- Gradio for the interactive interface
- The computer vision community for research and algorithms 