import cv2
import numpy as np
import os


def load_images_from_directory(pathfile):
    """
    Load all images from a directory and return them as a list.
    Ensures correct sorting and handles different image formats.
    """
    images = []
    filenames = sorted(os.listdir(pathfile))  # Ensure correct order

    for filename in filenames:
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            img_path = os.path.join(pathfile, filename)
            img = cv2.imread(
                img_path, cv2.IMREAD_COLOR
            )  # Load as color image for consistency
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Failed to load {img_path}")

    if len(images) < 2:
        raise ValueError("At least two valid images are required for stitching.")

    return images


def stitch_images_opencv(images):
    """
    Use OpenCV's built-in Stitcher module to stitch images.
    Returns the stitched image or None if stitching fails.
    """
    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        print(f"Image stitching failed with error code {status}")
        return None  # Return None instead of raising an error

    return stitched_image


# Load images
pathfile = "./dataset/mac"
try:
    images = load_images_from_directory(pathfile)

    # Stitch images
    result = stitch_images_opencv(images)
    output_filename = "stitched_output.jpg"
    cv2.imwrite(output_filename, result)
    print(f"Stitched image saved as {output_filename}")
    if result is not None:
        cv2.imshow("Stitched Image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Stitching failed. No output image generated.")
except Exception as e:
    print(f"Error: {e}")
