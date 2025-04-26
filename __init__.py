"""
Image Stitching for Panorama Generation package.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main functionality
from app import (
    analyze_image,
    detect_and_compute,
    match_features,
    find_homography,
    draw_matches,
    preprocess_image,
    stitch_images,
    process_images
) 