"""
- Runs independently in multiprocessing.Process
- Receives RGB images via multiprocessing.Queue
- Sends back detection results via another Queue
- Non-blocking design - main simulation continues smoothly
"""

import numpy as np
import cv2
import multiprocessing as mp
import time


def detect_red_object(rgb_image, camera_width, camera_height, debug=False):
    """
    Detect red object in RGB image and return its centroid position.
    
    Args:
        rgb_image: numpy array (H, W, 3) in RGB format
        camera_width: camera resolution width
        camera_height: camera resolution height
        debug: if True, prints detection info
    
    Returns:
        tuple: (pixel_x, pixel_y, detected, area)
            - pixel_x, pixel_y: centroid position in image coordinates (0,0 = top-left)
            - detected: boolean indicating if object was found
            - area: size of detected blob (pixels)
    """
    
    # Convert RGB to BGR for OpenCV
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    
    # Define red color range in HSV
    # Red wraps around in HSV (0-10 and 170-180), so we need two ranges
    lower_red1 = np.array([0, 100, 100])      # Lower red hues
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])    # Upper red hues
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine masks
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return camera_width / 2, camera_height / 2, False, 0
    
    # Find largest contour (assume it's our target)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Filter out tiny detections (noise)
    if area < 50:  # Minimum 50 pixels
        return camera_width / 2, camera_height / 2, False, 0
    
    # Compute centroid
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return camera_width / 2, camera_height / 2, False, 0
    
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])
    
    if debug:
        # Compute error from center
        error_x = centroid_x - camera_width / 2
        error_y = centroid_y - camera_height / 2
        print(f"[Vision] Red object detected at ({centroid_x}, {centroid_y}), "
              f"error: ({error_x:.1f}, {error_y:.1f}) px, area: {area:.0f} px²")
    
    return centroid_x, centroid_y, True, area


def vision_worker(image_queue, result_queue, camera_width, camera_height, debug=False):
    """
    Vision processing worker that runs in a separate process.
    
    Args:
        image_queue: multiprocessing.Queue for receiving RGB images
        result_queue: multiprocessing.Queue for sending detection results
        camera_width: camera resolution width
        camera_height: camera resolution height
        debug: if True, prints debug information
    """
    
    print("[Vision Worker] Started successfully")
    print(f"[Vision Worker] Camera resolution: {camera_width}x{camera_height}")
    print(f"[Vision Worker] Waiting for images...\n")
    
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Non-blocking get - process latest image only
            try:
                # Clear queue to get only the latest image
                rgb_image = None
                depth_array = None
                while not image_queue.empty():
                    item = image_queue.get_nowait()
                    if isinstance(item, tuple) and len(item) == 2:
                        rgb_image, depth_array = item
                    else:
                        rgb_image = item
                        depth_array = None

                if rgb_image is None:
                    time.sleep(0.01)
                    continue

                # Process image
                pixel_x, pixel_y, detected, area = detect_red_object(
                    rgb_image, camera_width, camera_height, debug=debug
                )

                # Get depth at detected centroid
                depth_value = None
                if detected and depth_array is not None:
                    # Clamp indices to valid range
                    px = int(np.clip(pixel_x, 0, camera_width - 1))
                    py = int(np.clip(pixel_y, 0, camera_height - 1))
                    depth_value = float(depth_array[py, px])

                # Send result (non-blocking)
                while not result_queue.empty():
                    try:
                        result_queue.get_nowait()
                    except:
                        break

                result_queue.put({
                    'pixel_x': pixel_x,
                    'pixel_y': pixel_y,
                    'detected': detected,
                    'area': area,
                    'depth': depth_value,
                    'timestamp': time.time()
                })
                
                frame_count += 1
                if detected:
                    detection_count += 1
                
                # Print statistics every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    detection_rate = (detection_count / frame_count) * 100 if frame_count > 0 else 0
                    print(f"[Vision Worker] Processed {frame_count} frames, "
                          f"{fps:.1f} FPS, {detection_rate:.1f}% detection rate")
                
            except Exception as e:
                if debug:
                    print(f"[Vision Worker] Processing error: {e}")
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n[Vision Worker] Shutting down...")


def start_vision_process(camera_width, camera_height, debug=False):
    """
    Start the vision processing worker in a separate process.
    
    Args:
        camera_width: camera resolution width
        camera_height: camera resolution height
        debug: if True, enables debug output
    
    Returns:
        tuple: (process, image_queue, result_queue)
            - process: multiprocessing.Process object
            - image_queue: Queue for sending images to worker
            - result_queue: Queue for receiving results from worker
    """
    
    # Create queues with limited size to prevent memory buildup
    image_queue = mp.Queue(maxsize=2)
    result_queue = mp.Queue(maxsize=2)
    
    # Create and start process
    process = mp.Process(
        target=vision_worker,
        args=(image_queue, result_queue, camera_width, camera_height, debug)
    )
    process.daemon = True  # Daemon process will exit when main process exits
    process.start()
    
    return process, image_queue, result_queue


# ============================================================================
# STANDALONE TESTING
# ============================================================================

def test_vision_processor():
    """Test vision processor with synthetic red image."""
    
    print("="*70)
    print("VISION PROCESSOR - STANDALONE TEST")
    print("="*70)
    
    # Create test image with red square
    width, height = 320, 240
    test_image = np.zeros((height, width, 3), dtype=np.uint8)
    test_image[:, :] = [128, 128, 128]  # Gray background
    
    # Draw red square at offset position
    center_x, center_y = 200, 150
    square_size = 40
    test_image[
        center_y - square_size//2 : center_y + square_size//2,
        center_x - square_size//2 : center_x + square_size//2
    ] = [255, 0, 0]  # Red square (RGB)
    
    print(f"Test image: {width}x{height}")
    print(f"Red square center: ({center_x}, {center_y})")
    print(f"Image center: ({width/2}, {height/2})")
    print(f"Expected error: ({center_x - width/2}, {center_y - height/2})\n")
    
    # Detect
    pixel_x, pixel_y, detected, area = detect_red_object(test_image, width, height, debug=True)
    
    print(f"\nDetection result:")
    print(f"  Detected: {detected}")
    print(f"  Position: ({pixel_x}, {pixel_y})")
    print(f"  Area: {area} px²")
    print(f"  Error from center: ({pixel_x - width/2:.1f}, {pixel_y - height/2:.1f}) px")
    
    if detected and abs(pixel_x - center_x) < 5 and abs(pixel_y - center_y) < 5:
        print("\n✓ Test PASSED - Detection accurate!")
    else:
        print("\n✗ Test FAILED - Detection error too large")


if __name__ == "__main__":
    test_vision_processor()
