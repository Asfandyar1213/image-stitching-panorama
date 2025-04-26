"""
Example script demonstrating how to use the image stitching functionality programmatically.
This allows integration into other applications without using the Gradio interface.
"""

import cv2
import numpy as np
import argparse
import sys
import os
from datetime import datetime

# Import functions from app.py
from app import (
    analyze_image,
    detect_and_compute,
    match_features,
    find_homography,
    draw_matches,
    preprocess_image,
    stitch_images
)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Stitching Example")
    parser.add_argument("--image1", required=True, help="Path to first input image")
    parser.add_argument("--image2", required=True, help="Path to second input image")
    parser.add_argument("--output", default="output_panorama.jpg", help="Path to output panorama image")
    parser.add_argument("--blend", choices=["average", "multiband"], default="average", help="Blending method")
    parser.add_argument("--show", action="store_true", help="Display results")
    parser.add_argument("--adjust", action="store_true", help="Auto-adjust image parameters")
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.image1):
        print(f"Error: First image file '{args.image1}' does not exist")
        sys.exit(1)
    if not os.path.exists(args.image2):
        print(f"Error: Second image file '{args.image2}' does not exist")
        sys.exit(1)
    
    # Load images
    print(f"Loading images: {args.image1} and {args.image2}")
    img1 = cv2.imread(args.image1)
    img2 = cv2.imread(args.image2)
    
    if img1 is None or img2 is None:
        print("Error: Failed to load one or both images")
        sys.exit(1)
    
    # Preprocess images
    if args.adjust:
        print("Analyzing images and adjusting parameters...")
        stats1 = analyze_image(img1)
        stats2 = analyze_image(img2)
        
        # Average the optimal parameters from both images
        brightness = (stats1['brightness'] + stats2['brightness']) / 2
        contrast = (stats1['contrast'] + stats2['contrast']) / 2
        blur = max(stats1['blur'], stats2['blur'])
        
        print(f"Auto-adjusting parameters: brightness={brightness:.1f}, contrast={contrast:.1f}, blur={blur}")
        img1 = preprocess_image(img1, brightness, contrast, blur)
        img2 = preprocess_image(img2, brightness, contrast, blur)
    
    # Stitch images
    print(f"Stitching images using {args.blend} blending...")
    result, matches_vis, message = stitch_images(img1, img2, args.blend)
    
    # Display results
    if args.show and matches_vis is not None:
        print("Displaying matches...")
        cv2.imshow("Feature Matches", matches_vis)
        cv2.waitKey(1500)  # Show matches briefly
    
    # Process result
    if result is not None:
        print(f"Stitching successful: {message}")
        
        # Save output image
        print(f"Saving panorama to {args.output}")
        cv2.imwrite(args.output, result)
        
        # Display result if requested
        if args.show:
            cv2.imshow("Panorama", result)
            print("Press any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(f"Stitching failed: {message}")
        if matches_vis is not None and args.show:
            print("Press any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        sys.exit(1)
    
    print("Done!")

if __name__ == "__main__":
    main() 