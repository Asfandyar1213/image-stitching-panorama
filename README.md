# Image Stitching for Panorama Generation

![Panorama Banner](https://i.imgur.com/example.png)

## Overview

This project implements an advanced image stitching system to create panoramas from multiple images. It uses computer vision techniques including SIFT feature detection, FLANN-based feature matching, and homography transformation to seamlessly combine images.

### Key Features

- **Intelligent Image Analysis**: Automatically analyzes and adjusts image parameters
- **Advanced Feature Detection**: Scale-Invariant Feature Transform (SIFT)
- **Robust Feature Matching**: FLANN-based matching with ratio test
- **Multi-method Blending**: Average and multi-band blending options
- **Interactive UI**: User-friendly Gradio interface
- **Error Visualization**: Shows feature matches for better debugging
- **Parameter Customization**: Manual controls for brightness, contrast, and blur

## Installation

### Prerequisites

- Python 3.7+
- OpenCV
- NumPy
- Gradio

### Setup

1. Clone this repository:
```bash
git clone https://github.com/YourUsername/image-stitching-panorama.git
cd image-stitching-panorama
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to the URL shown (typically http://127.0.0.1:7860)

3. Use the interface to:
   - Upload two images with overlapping regions
   - Adjust processing parameters if needed
   - Generate and view the panorama
   - Save the result

### Example Usage

The application provides controls for:
- **Brightness Adjustment**: Enhance dark or bright images
- **Contrast Adjustment**: Improve feature detection in low-contrast images
- **Blur Amount**: Reduce noise in sharper images
- **Blending Method**: Choose between average and multi-band blending

## How It Works

The image stitching process follows these steps:

1. **Image Analysis**: Analyze brightness, contrast, and sharpness
2. **Parameter Adjustment**: Automatically adjust processing parameters
3. **Feature Detection**: Find distinctive points using SIFT
4. **Feature Matching**: Match points between images using FLANN
5. **Homography Estimation**: Calculate transformation matrix
6. **Image Warping**: Transform and align images
7. **Image Blending**: Seamlessly combine images

## Technical Details

### Adaptive Processing

The system adapts to different image characteristics:
- For dark images: Increases brightness
- For low-contrast images: Enhances contrast
- For noisy images: Applies appropriate blur

### Error Handling

Comprehensive error handling for common issues:
- Insufficient features
- Poor matches
- Invalid homography
- Processing errors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV library for computer vision algorithms
- Gradio for the interactive interface 