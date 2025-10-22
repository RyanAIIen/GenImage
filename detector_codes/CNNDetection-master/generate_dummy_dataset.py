import os
from PIL import Image, ImageDraw
import numpy as np


def create_real_image(size=(224, 224)):
    # Create simple patterns for "real" images
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)

    # Draw simple rectangles with solid colors
    color = tuple(np.random.randint(0, 256, 3))
    draw.rectangle([50, 50, 174, 174], fill=color)

    return img


def create_fake_image(size=(224, 224)):
    # Create more complex patterns for "fake" images
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)

    # Draw multiple shapes with gradients and patterns
    for _ in range(5):
        color = tuple(np.random.randint(0, 256, 3))
        x1 = np.random.randint(0, size[0])
        y1 = np.random.randint(0, size[1])
        x2 = np.random.randint(x1, size[0])
        y2 = np.random.randint(y1, size[1])
        draw.ellipse([x1, y1, x2, y2], fill=color)

    return img


def generate_dataset(base_path, num_images=100):
    splits = ['train', 'val', 'test']
    categories = ['generic']

    for split in splits:
        for category in categories:
            # Create directory structure
            real_dir = os.path.join(base_path, split, category, '0_real')
            fake_dir = os.path.join(base_path, split, category, '1_fake')
            os.makedirs(real_dir, exist_ok=True)
            os.makedirs(fake_dir, exist_ok=True)

            # Generate images
            for i in range(num_images):
                # Generate and save real image
                real_img = create_real_image()
                real_img.save(os.path.join(real_dir, f'real_{i:03d}.png'))

                # Generate and save fake image
                fake_img = create_fake_image()
                fake_img.save(os.path.join(fake_dir, f'fake_{i:03d}.png'))


if __name__ == '__main__':
    base_path = 'dataset'
    generate_dataset(base_path)
    print(f"Generated dummy dataset in {base_path}/")
    print("Structure:")
    print("dataset/")
    print("├── train/")
    print("│   └── generic/")
    print("│       ├── 0_real/ (100 images)")
    print("│       └── 1_fake/ (100 images)")
    print("├── val/")
    print("│   └── generic/")
    print("│       ├── 0_real/ (100 images)")
    print("│       └── 1_fake/ (100 images)")
    print("└── test/")
    print("    └── generic/")
    print("        ├── 0_real/ (100 images)")
    print("        └── 1_fake/ (100 images)")
