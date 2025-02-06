import cv2
import numpy as np
import os

def load_images_from_directory(pathfile):
    images = []
    filenames = sorted(os.listdir(pathfile))  # Ensure correct order

    for filename in filenames:
        if filename.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(pathfile, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Handle different formats
            if img is not None:
                images.append(img)
    
    if len(images) < 2:
        raise ValueError("At least two images are required for stitching.")

    return images

def stitch_images_opencv(images):
    """
    Use OpenCV's built-in Stitcher module for efficient stitching.
    """
    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"Image stitching failed with error code {status}")

    return stitched_image

# Load images
pathfile = "./dataset/eclipse/"
images = load_images_from_directory(pathfile)

# Stitch images
try:
    result = stitch_images_opencv(images)
    cv2.imshow("Stitched Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as e:
    print(f"Error: {e}")
