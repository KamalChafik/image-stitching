import cv2
import numpy as np
import os as os

def load_images_from_directory(pathfile):
    images = []
    for filename in os.listdir(pathfile):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(pathfile, filename))
            if img is not None:
                images.append(img)
    return images

def stitch_images(images):
    # Convert images to grayscale
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    keypoints, descriptors = [], []
    for img in gray_images:
        kp, des = sift.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)

    # FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find matches between first and second image
    matches = flann.knnMatch(descriptors[0], descriptors[1], k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract matched points
    src_pts = np.float32([keypoints[0][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints[1][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute Homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp first image
    h1, w1 = images[0].shape[:2]
    h2, w2 = images[1].shape[:2]
    result = cv2.warpPerspective(images[0], H, (w1 + w2, h1))

    # Paste the second image onto the result
    result[0:h2, 0:w2] = images[1]

    return result


# Load images from directory
pathfile = "./dataset/flower"
images = load_images_from_directory(pathfile)

# Stitch images
result = stitch_images(images)

# Show result
cv2.imshow("Stitched Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
