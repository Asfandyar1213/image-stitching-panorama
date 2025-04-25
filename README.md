# Image Stitching for Panorama Generation

This project implements image stitching to create panoramas from multiple images using OpenCV and Gradio.

## Features

- Feature detection using SIFT
- Feature matching using FLANN matcher
- Homography transformation
- Image blending for seamless panoramas
- User-friendly Gradio interface

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Gradio

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://127.0.0.1:7860)

3. Upload two images that you want to stitch together
   - The images should have some overlapping areas for better results
   - The images should be taken from the same position with different angles

4. Click "Submit" to generate the panorama

## How it Works

1. The application detects keypoints and descriptors in both images using SIFT
2. It matches the features between the two images
3. Computes the homography matrix to transform one image to align with the other
4. Warps and blends the images to create a seamless panorama

## Notes

- For best results, use images with good overlap (30-50%)
- Images should be taken from the same position
- Avoid moving objects in the scene
- Good lighting conditions help in feature detection 