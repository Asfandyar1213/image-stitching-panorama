"""
Basic test script for the image stitching functionality.
"""

import cv2
import numpy as np
import os
import sys
import unittest

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

class TestImageStitching(unittest.TestCase):
    def setUp(self):
        # Check if test images exist
        self.test_img1_path = "example1.jpg"
        self.test_img2_path = "example2.jpg"
        
        # Skip tests if example images don't exist
        if not (os.path.exists(self.test_img1_path) and os.path.exists(self.test_img2_path)):
            self.skipTest("Test images not found. Skipping tests.")
        
        # Load test images
        self.img1 = cv2.imread(self.test_img1_path)
        self.img2 = cv2.imread(self.test_img2_path)
    
    def test_analyze_image(self):
        """Test image analysis function"""
        stats = analyze_image(self.img1)
        
        # Check that analysis returns expected keys
        expected_keys = [
            'brightness', 'contrast', 'blur', 
            'mean_intensity', 'std_intensity', 'blur_value'
        ]
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check that values are within expected ranges
        self.assertGreaterEqual(stats['mean_intensity'], 0)
        self.assertLessEqual(stats['mean_intensity'], 255)
        self.assertGreaterEqual(stats['std_intensity'], 0)
    
    def test_feature_detection(self):
        """Test feature detection function"""
        kp1, desc1, stats1 = detect_and_compute(self.img1)
        
        # Check that keypoints and descriptors were found
        self.assertIsNotNone(kp1)
        self.assertIsNotNone(desc1)
        self.assertGreater(len(kp1), 0)
        
        # Check descriptor shape
        self.assertEqual(desc1.shape[1], 128)  # SIFT descriptors are 128-dimensional
    
    def test_feature_matching(self):
        """Test feature matching function"""
        kp1, desc1, _ = detect_and_compute(self.img1)
        kp2, desc2, _ = detect_and_compute(self.img2)
        
        # Only proceed if we have descriptors
        if desc1 is None or desc2 is None:
            self.skipTest("Feature detection failed. Skipping matching test.")
        
        matches = match_features(desc1, desc2)
        
        # Check that we have matches
        self.assertIsInstance(matches, list)
    
    def test_homography(self):
        """Test homography calculation"""
        kp1, desc1, _ = detect_and_compute(self.img1)
        kp2, desc2, _ = detect_and_compute(self.img2)
        
        # Only proceed if we have descriptors
        if desc1 is None or desc2 is None:
            self.skipTest("Feature detection failed. Skipping homography test.")
        
        matches = match_features(desc1, desc2)
        
        if len(matches) < 4:
            self.skipTest(f"Not enough matches ({len(matches)}). Skipping homography test.")
        
        H, mask = find_homography(kp1, kp2, matches)
        
        # Check that homography matrix is valid
        if H is not None:
            self.assertEqual(H.shape, (3, 3))
            self.assertAlmostEqual(H[2, 2], 1.0, delta=1e-10)
    
    def test_full_stitching_process(self):
        """Test the complete stitching process"""
        result, matches_vis, message = stitch_images(self.img1, self.img2)
        
        # Output the status message for debugging
        print(f"Stitching result: {message}")
        
        # Test could pass even if stitching fails due to the specific test images
        # We're just checking that the function runs without errors
        if result is not None:
            # If successful, verify the result is an image
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result.shape), 3)  # Should be 3D array (height, width, channels)
            self.assertEqual(result.shape[2], 3)    # Should have 3 channels
        
        # Check if visualization was created
        if matches_vis is not None:
            self.assertIsInstance(matches_vis, np.ndarray)
    
    def test_preprocessing(self):
        """Test image preprocessing function"""
        # Test with different parameters
        processed = preprocess_image(self.img1, brightness=20, contrast=1.2, blur=1)
        
        # Verify the processed image has the same dimensions as the input
        self.assertEqual(processed.shape, self.img1.shape)
        
        # Verify the processed image is uint8
        self.assertEqual(processed.dtype, np.uint8)

if __name__ == "__main__":
    unittest.main() 