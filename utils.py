import os
import math
import random
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image

dataset_dir = "CatFLW dataset"


def show_random_cat_labels(total: int) -> None:
    if total <= 0:
        return
    y_tiles = math.ceil(math.sqrt(total))
    x_tiles = math.ceil(total / y_tiles)

    imgs_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    # Get all images with supported extensions
    all_imgs = [
        f for f in os.listdir(imgs_dir)
        if os.path.splitext(f)[1].lower().lstrip('.') in ['png', 'jpg', 'jpeg', 'bmp']
    ]

    # Filter to only images that have a matching label file
    valid_imgs = []
    for img in all_imgs:
        stem = os.path.splitext(img)[0]
        label_file = f"{stem}.json"
        if os.path.exists(os.path.join(labels_dir, label_file)):
            valid_imgs.append(img)

    if not valid_imgs:
        print("No matching image and label pairs found.")
        return

    # Select random images
    if len(valid_imgs) >= total:
        rand_imgs = random.sample(valid_imgs, total)
    else:
        rand_imgs = random.choices(valid_imgs, k=total)

    # Map directly to ensure they match at each index
    rand_labels = [f"{os.path.splitext(img)[0]}.json" for img in rand_imgs]

    fig, axes = plt.subplots(x_tiles, y_tiles, figsize=(15, 15), squeeze=False)
    axes = axes.flatten()

    i = -1
    for i in range(len(rand_imgs)):
        ax = axes[i]
        
        label_path = os.path.join(labels_dir, rand_labels[i])
        with open(label_path, 'r') as file:
            data = json.load(file)

        landmarks = data["labels"]  # List of 48 points [x, y]
        bbox = data["bounding_boxes"]  # [left, top, right, bottom]

        stem = os.path.splitext(rand_imgs[i])[0]
        img = Image.open(os.path.join(imgs_dir, rand_imgs[i]))
        ax.imshow(img)

        xs = [pt[0] for pt in landmarks]
        ys = [pt[1] for pt in landmarks]
        ax.scatter(xs, ys, c='red', s=2, marker='o', alpha=0.8, edgecolors='none')

        left, top, right, bottom = bbox
        rect = patches.Rectangle(
            (left, top), right - left, bottom - top,
            linewidth=1, edgecolor='cyan', facecolor='none'
        )
        ax.add_patch(rect)
        
        ax.axis('off')
        ax.set_title(stem, fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    show_random_cat_labels(15)