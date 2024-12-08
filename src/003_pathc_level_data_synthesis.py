import os
import random
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from configuration import synthetic_fore_crop, synthetic_fore_weed, synthetic_back_soil, SYNTHETIC_PATH
from configuration import class_names


class_indices = {name: i for i, name in enumerate(class_names)}

def get_random_image(folder):
    files = os.listdir(folder)
    filename = random.choice(files)
    img = Image.open(os.path.join(folder, filename)).convert("RGB")
    return img

def get_random_individual(folder):
    files = os.listdir(folder)
    filename = random.choice(files)
    img = Image.open(os.path.join(folder, filename)).convert("RGBA")
    return img, os.path.join(folder, filename)

def paste_randomly(background, image, image_mask, mask):
    width, height = background.size

    max_x = width - image.size[0]
    max_y = height - image.size[1]

    if max_x < 0:
        image = image.crop((0, 0, width, image.size[1]))
        image_mask = image_mask.crop((0, 0, width, image_mask.size[1]))
        max_x = 0
    
    if max_y < 0:
        image = image.crop((0, 0, image.size[0], height))
        image_mask = image_mask.crop((0, 0, image_mask.size[0], height))
        max_y = 0

    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    background.paste(image, (x, y))

    image_mask_np = np.array(image_mask)
    updated_image_mask = Image.fromarray(image_mask_np)
    mask.paste(updated_image_mask, (x, y))

def create_synthetic_scene(idx):
    random.seed(42 + idx)
    np.random.seed(42 + idx)
    
    soil_image = get_random_image(synthetic_back_soil)
    soil_image = soil_image.copy().convert("RGB")
    mask = Image.new("L", soil_image.size, class_indices['soil'])

    crop_images = []
    for _ in range(random.randint(1, 3)):
        crop_type = random.choice(os.listdir(synthetic_fore_crop))
        image, image_mask = get_random_individual(os.path.join(synthetic_fore_crop, crop_type, 'image'))
        mask_path = os.path.join(synthetic_fore_crop, crop_type, 'mask', Path(image_mask).name)
        image_mask = Image.open(mask_path).convert("L")
        crop_images.append((image, image_mask, crop_type))

    weed_images = []
    for _ in range(random.randint(1, 5)):
        weed_type = random.choice(os.listdir(synthetic_fore_weed))
        image, image_mask = get_random_individual(os.path.join(synthetic_fore_weed, weed_type, 'image'))
        mask_path = os.path.join(synthetic_fore_weed, weed_type, 'mask', Path(image_mask).name)
        image_mask = Image.open(mask_path).convert("L")
        weed_images.append((image, image_mask, weed_type))

    for image, image_mask, _ in crop_images:
        paste_randomly(soil_image, image, image_mask, mask)

    for image, image_mask, _ in weed_images:
        paste_randomly(soil_image, image, image_mask, mask)

    soil_image.save(os.path.join(image_output_dir, f"synthetic_{idx}.png"))
    mask.save(os.path.join(mask_output_dir, f"synthetic_{idx}.png"))

def generate_scenes(num_scenes):
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(create_synthetic_scene, range(num_scenes)), total=num_scenes))
            
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic patch-level data.")
    parser.add_argument("-x", type=int, default=1, help="Multiplier for the number of synthetic scenes.")
    args = parser.parse_args()

    # 设置生成的倍数和输出目录
    multiplier = args.x
    output_dir = f"{SYNTHETIC_PATH}/synthetic_patch_{multiplier}x"
    num_scenes = 1540 * multiplier

    print(f"Generating {num_scenes} synthetic scenes in '{output_dir}'...")
    random.seed(42)
    np.random.seed(42)