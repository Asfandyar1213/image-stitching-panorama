"""
Demo script that stitches multiple images in sequence to create a larger panorama.
Usage: python demo.py --dir <directory_with_images> --output <output_file> --blend <blending_method>
"""

import cv2
import numpy as np
import os
import argparse
import glob
from tqdm import tqdm

# Import functions from app.py
from app import stitch_images, preprocess_image, analyze_image

def stitch_multiple_images(images, blend_method="average"):
    """
    Stitch multiple images in sequence.
    
    Args:
        images: List of images to stitch
        blend_method: Blending method to use
        
    Returns:
        Stitched panorama or None if stitching failed
        Status message
    """
    if len(images) < 2:
        return None, "At least two images are required for stitching"
    
    print(f"Starting panorama creation with {len(images)} images...")
    
    # Start with the first image as our panorama
    panorama = images[0]
    success_count = 0
    messages = []
    
    # Process each image in sequence
    for i, image in enumerate(tqdm(images[1:], desc="Stitching images")):
        print(f"\nProcessing image {i+1}/{len(images)-1}...")
        
        # Stitch current panorama with next image
        result, _, message = stitch_images(panorama, image, blend_method)
        messages.append(message)
        
        if result is not None:
            # Update panorama with new stitched result
            panorama = result
            success_count += 1
            print(f"Successfully added image {i+1}")
        else:
            print(f"Failed to add image {i+1}: {message}")
    
    # Return final panorama and summary
    if success_count == 0:
        return None, "Failed to stitch any images"
    
    summary = f"Successfully stitched {success_count}/{len(images)-1} images"
    return panorama, summary

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-image Panorama Demo")
    parser.add_argument("--dir", required=True, help="Directory containing images to stitch")
    parser.add_argument("--pattern", default="*.jpg", help="Pattern for image files (default: *.jpg)")
    parser.add_argument("--output", default="panorama_result.jpg", help="Output panorama filename")
    parser.add_argument("--blend", choices=["average", "multiband"], default="average", help="Blending method")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess images before stitching")
    parser.add_argument("--sort", action="store_true", help="Sort images by filename")
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.isdir(args.dir):
        print(f"Error: Directory '{args.dir}' does not exist")
        return 1
    
    # Get list of image files
    image_files = glob.glob(os.path.join(args.dir, args.pattern))
    
    if len(image_files) < 2:
        print(f"Error: Found {len(image_files)} images. At least 2 are required.")
        return 1
    
    # Sort images if requested
    if args.sort:
        image_files.sort()
    
    print(f"Found {len(image_files)} images: {', '.join(os.path.basename(f) for f in image_files)}")
    
    # Load images
    images = []
    for img_path in image_files:
        print(f"Loading {os.path.basename(img_path)}...")
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not load {img_path}, skipping")
            continue
            
        # Preprocess if requested
        if args.preprocess:
            print(f"Preprocessing {os.path.basename(img_path)}...")
            stats = analyze_image(img)
            img = preprocess_image(
                img, 
                brightness=stats['brightness'],
                contrast=stats['contrast'],
                blur=stats['blur']
            )
            
        images.append(img)
    
    if len(images) < 2:
        print("Error: Not enough valid images to create panorama")
        return 1
    
    # Stitch images
    panorama, message = stitch_multiple_images(images, args.blend)
    
    # Save result if successful
    if panorama is not None:
        print(f"\nStitching complete: {message}")
        print(f"Saving panorama to {args.output}")
        cv2.imwrite(args.output, panorama)
        
        # Display image size
        height, width = panorama.shape[:2]
        print(f"Panorama dimensions: {width}x{height} pixels")
        
        # Display result
        scale_factor = min(1.0, 1200 / width)
        if scale_factor < 1.0:
            display_img = cv2.resize(panorama, (0, 0), fx=scale_factor, fy=scale_factor)
        else:
            display_img = panorama
            
        cv2.imshow("Panorama Result", display_img)
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return 0
    else:
        print(f"\nFailed to create panorama: {message}")
        return 1

if __name__ == "__main__":
    main() 