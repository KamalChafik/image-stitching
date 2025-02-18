import os
import re

import matplotlib.pyplot as plt
from PIL import Image

folder_paths = ["dataset/eclipse", "dataset/flower", "dataset/niagara", "dataset/mac"]

for folder_path in folder_paths:
    image_paths = sorted(
        [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if re.search(r"\.(jpe?g|png)$", file)
        ]
    )

    # Open images and store them in a list
    images = [Image.open(image_path) for image_path in image_paths]

    # Calculate the number of rows needed (3 images per row)
    num_images = len(images)
    num_rows = (num_images + 2) // 3

    # Create a figure with a fixed size (1920x1080 pixels at dpi=100 equals 19.2x10.8 inches)
    fig, ax = plt.subplots(num_rows, 3, figsize=(19.2, 10.8), squeeze=False)
    fig.suptitle(os.path.basename(folder_path).capitalize(), fontsize=40)

    # Display each image in the subplot
    for i, image in enumerate(images):
        row = i // 3
        col = i % 3
        ax[row, col].imshow(image)
        ax[row, col].axis("off")
        ax[row, col].set_title(f"{i + 1}", fontsize=20)

    # Hide any unused subplots
    for j in range(num_images, num_rows * 3):
        row = j // 3
        col = j % 3
        ax[row, col].axis("off")

    # Save the figure at 1080p by specifying dpi
    plt.savefig(f"{folder_path}_combined_image.jpg", dpi=100)
    plt.close(fig)
