import cv2
import numpy as np
import gradio as gr
import os
from datetime import datetime

def analyze_image(image):
    """Analyze image characteristics and return optimal parameters"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate image statistics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Determine optimal parameters
    brightness = 0
    contrast = 1.0
    blur = 0
    
    # Adjust brightness if image is too dark or bright
    if mean_intensity < 100:
        brightness = 20
    elif mean_intensity > 200:
        brightness = -20
    
    # Adjust contrast if image has low contrast
    if std_intensity < 50:
        contrast = 1.2
    
    # Add blur if image is too sharp
    if blur_value > 500:
        blur = 1
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'blur': blur,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'blur_value': blur_value
    }

def detect_and_compute(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Analyze image characteristics
    stats = analyze_image(image)
    
    # Adjust SIFT parameters based on image characteristics
    if stats['blur_value'] > 500:
        # For sharp images, use more features
        sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=4, contrastThreshold=0.03, edgeThreshold=12, sigma=1.6)
    elif stats['std_intensity'] < 50:
        # For low contrast images, use more sensitive parameters
        sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=8, sigma=1.4)
    else:
        # Default parameters
        sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    
    # Find keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors, stats

def match_features(desc1, desc2):
    # FLANN matcher with more trees and checks
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(desc1, desc2, k=2)
    except Exception as e:
        return [], f"Error in feature matching: {str(e)}"
    
    # Apply ratio test with more lenient threshold
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:  # More lenient ratio
            good_matches.append(m)
    return good_matches

def find_homography(kp1, kp2, matches):
    if len(matches) < 4:
        return None, f"Not enough matches (found {len(matches)}, need at least 4)"
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Use RANSAC with more iterations and lower threshold
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0, maxIters=2000)
    
    if H is None:
        return None, "Homography computation failed"
    
    # Check if homography is valid
    if np.abs(H[2, 2]) < 1e-10:
        return None, "Invalid homography matrix"
    
    # Ensure H is float32
    H = H.astype(np.float32)
    
    return H, mask

def draw_matches(image1, image2, kp1, kp2, matches):
    # Create a new image with both images side by side
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = image1
    vis[:h2, w1:w1 + w2] = image2
    
    # Draw matches
    for match in matches:
        pt1 = (int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1]))
        pt2 = (int(kp2[match.trainIdx].pt[0] + w1), int(kp2[match.trainIdx].pt[1]))
        cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(vis, pt1, 3, (0, 0, 255), -1)
        cv2.circle(vis, pt2, 3, (0, 0, 255), -1)
    
    return vis

def preprocess_image(image, brightness=0, contrast=1.0, blur=0):
    # Convert to float32 for processing
    img = image.astype(np.float32)
    
    # Apply brightness adjustment
    img = img + brightness
    
    # Apply contrast adjustment
    img = (img - 128) * contrast + 128
    
    # Apply Gaussian blur if needed
    if blur > 0:
        img = cv2.GaussianBlur(img, (2*blur+1, 2*blur+1), 0)
    
    # Clip values to valid range
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def stitch_images(image1, image2, blend_method="average"):
    try:
        # Detect features with automatic parameter adjustment
        kp1, desc1, stats1 = detect_and_compute(image1)
        kp2, desc2, stats2 = detect_and_compute(image2)
        
        if desc1 is None or desc2 is None:
            return None, None, "Feature detection failed"
        
        if len(kp1) < 4 or len(kp2) < 4:
            return None, None, f"Not enough features detected (Image 1: {len(kp1)}, Image 2: {len(kp2)})"
        
        # Match features
        matches = match_features(desc1, desc2)
        
        if len(matches) < 4:
            return None, None, f"Not enough good matches found ({len(matches)} matches)"
        
        # Draw matches for visualization
        matches_vis = draw_matches(image1, image2, kp1, kp2, matches)
        
        # Find homography
        H, mask = find_homography(kp1, kp2, matches)
        if H is None:
            return None, matches_vis, "Could not compute homography"
        
        # Warp image
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        
        # Get the corners of the first image
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        
        # Transform corners of first image
        corners1_transformed = cv2.perspectiveTransform(corners1, H)
        
        # Find the dimensions of the output image
        all_corners = np.concatenate((corners2, corners1_transformed), axis=0)
        x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # Translation
        translation = [-x_min, -y_min]
        H_translation = np.array([
            [1, 0, translation[0]],
            [0, 1, translation[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Combine homography and translation
        H_combined = H_translation.dot(H)
        
        # Ensure H_combined is float32
        H_combined = H_combined.astype(np.float32)
        
        # Warp images
        output_shape = (y_max - y_min, x_max - x_min)
        try:
            warped_img1 = cv2.warpPerspective(image1, H_combined, (output_shape[1], output_shape[0]))
            warped_img2 = cv2.warpPerspective(image2, H_translation, (output_shape[1], output_shape[0]))
        except Exception as e:
            return None, matches_vis, f"Error during warping: {str(e)}"
        
        # Blend images based on method
        if blend_method == "average":
            # Simple averaging
            mask1 = (warped_img1 != 0).astype(np.float32)
            mask2 = (warped_img2 != 0).astype(np.float32)
            overlap = mask1 * mask2
            blended = warped_img1.copy()
            blended[overlap > 0] = (warped_img1[overlap > 0] + warped_img2[overlap > 0]) / 2
            blended[mask2 > 0] = warped_img2[mask2 > 0]
        elif blend_method == "multiband":
            # Multi-band blending (simplified version)
            mask1 = (warped_img1 != 0).astype(np.float32)
            mask2 = (warped_img2 != 0).astype(np.float32)
            overlap = mask1 * mask2
            
            # Create distance maps
            dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
            dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)
            
            # Normalize distance maps
            dist1 = dist1 / (dist1.max() + 1e-7)
            dist2 = dist2 / (dist2.max() + 1e-7)
            
            # Create weight maps
            weight1 = dist1 / (dist1 + dist2 + 1e-7)
            weight2 = dist2 / (dist1 + dist2 + 1e-7)
            
            # Blend images
            blended = (warped_img1 * weight1[..., np.newaxis] + 
                      warped_img2 * weight2[..., np.newaxis]).astype(np.uint8)
        
        return blended, matches_vis, "Success"
    except Exception as e:
        return None, None, f"Error during stitching: {str(e)}"

def process_images(image1, image2, brightness=0, contrast=1.0, blur=0, blend_method="average"):
    if image1 is None or image2 is None:
        return None, None, "Please upload both images"
    
    try:
        # Convert images to numpy arrays
        img1 = np.array(image1)
        img2 = np.array(image2)
        
        # Convert to BGR (OpenCV format)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        
        # Analyze images and get optimal parameters
        stats1 = analyze_image(img1)
        stats2 = analyze_image(img2)
        
        # Use the average of optimal parameters from both images
        brightness = (stats1['brightness'] + stats2['brightness']) / 2
        contrast = (stats1['contrast'] + stats2['contrast']) / 2
        blur = max(stats1['blur'], stats2['blur'])
        
        # Preprocess images
        img1 = preprocess_image(img1, brightness, contrast, blur)
        img2 = preprocess_image(img2, brightness, contrast, blur)
        
        # Stitch images
        result, matches_vis, message = stitch_images(img1, img2, blend_method)
        
        if result is None:
            return None, matches_vis, message
        
        # Convert back to RGB for display
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        if matches_vis is not None:
            matches_vis = cv2.cvtColor(matches_vis, cv2.COLOR_BGR2RGB)
        return result, matches_vis, message
    except Exception as e:
        return None, None, f"Error processing images: {str(e)}"

def save_result(image):
    if image is None:
        return None, "No image to save"
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/panorama_{timestamp}.jpg"
        
        # Save image
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return filename, "Image saved successfully"
    except Exception as e:
        return None, f"Error saving image: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Image Stitching for Panorama Generation") as iface:
    gr.Markdown("# Image Stitching for Panorama Generation")
    gr.Markdown("Upload two images to create a panorama. Adjust preprocessing parameters for better results.")
    
    with gr.Row():
        with gr.Column():
            image1 = gr.Image(label="First Image", type="numpy")
            image2 = gr.Image(label="Second Image", type="numpy")
        
        with gr.Column():
            with gr.Group():
                gr.Markdown("### Preprocessing Parameters")
                brightness = gr.Slider(-50, 50, 0, label="Brightness Adjustment")
                contrast = gr.Slider(0.5, 2.0, 1.0, label="Contrast Adjustment")
                blur = gr.Slider(0, 5, 0, label="Blur Amount")
                blend_method = gr.Radio(["average", "multiband"], label="Blending Method", value="average")
    
    with gr.Row():
        process_btn = gr.Button("Generate Panorama", variant="primary")
        save_btn = gr.Button("Save Result", variant="secondary")
    
    with gr.Row():
        output_image = gr.Image(label="Stitched Panorama", type="numpy")
        matches_image = gr.Image(label="Feature Matches", type="numpy")
    
    status_message = gr.Textbox(label="Status", interactive=False)
    
    # Set up event handlers
    process_btn.click(
        fn=process_images,
        inputs=[image1, image2, brightness, contrast, blur, blend_method],
        outputs=[output_image, matches_image, status_message]
    )
    
    save_btn.click(
        fn=save_result,
        inputs=output_image,
        outputs=[status_message, status_message]
    )

if __name__ == "__main__":
    iface.launch(share=True) 