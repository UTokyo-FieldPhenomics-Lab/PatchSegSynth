import os
import imageio.v3 as iio
from pathlib import Path
import numpy as np
from skimage import img_as_ubyte
from skimage.measure import label, regionprops
from concurrent.futures import ThreadPoolExecutor
# from configs import *
from configuration import crop_species, weed_species, class_names
from configuration import real_train_images, real_train_annotations

from configuration import synthetic_fore_crop, synthetic_fore_weed, synthetic_back_soil

def create_directories():
    # create directories
    for crop in crop_species:
        os.makedirs(os.path.join(synthetic_fore_crop, crop.replace(' ', '_')), exist_ok=True)
    for weed in weed_species:
        os.makedirs(os.path.join(synthetic_fore_weed, weed.replace(' ', '_')), exist_ok=True)
    os.makedirs(synthetic_back_soil, exist_ok=True)
    
def save_cropped_image(image, mask, current_mask,\
                       image_name, class_name, output_dir, idx, min_area=100):
    image_dir = os.path.join(output_dir, class_name, 'image')
    mask_dir = os.path.join(output_dir, class_name, 'mask')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    labeled_mask = label(current_mask)
    regions = regionprops(labeled_mask)
    
    image_height, image_width = image.shape[:2]
    
    for i, region in enumerate(regions):
        if region.area >= min_area:
            minr, minc, maxr, maxc = region.bbox
            if minr == 0 or minc == 0 or maxr == image_height or maxc == image_width:
                continue
            cropped_image = image[minr:maxr, minc:maxc, :]
            cropped_mask = mask[minr:maxr, minc:maxc]
            
            file_name = f"{class_name}_{image_name[:-4]}_{idx}_{i}.png"
            iio.imwrite(os.path.join(image_dir, file_name), img_as_ubyte(cropped_image))
            iio.imwrite(os.path.join(mask_dir, file_name), cropped_mask)

def maximal_rectangle(matrix):
    if not matrix.size:
        return (0, 0, 0, 0)

    rows, cols = matrix.shape
    height = np.zeros(cols, dtype=int)  # 高度数组
    best_rectangle = (0, 0, 0, 0)       # (top, left, bottom, right)
    max_area = 0

    for i in range(rows):
        # 更新高度数组
        for j in range(cols):
            height[j] = height[j] + 1 if matrix[i][j] == 1 else 0

        # 使用单调栈找当前行的最大矩形
        stack = []
        for j in range(cols + 1):
            # 在最后插入一个高度为0，确保栈中元素全部出栈
            current_height = height[j] if j < cols else 0
            while stack and current_height < height[stack[-1]]:
                h = height[stack.pop()]
                left = stack[-1] + 1 if stack else 0
                right = j
                area = h * (right - left)
                if area > max_area:
                    max_area = area
                    best_rectangle = (i - h + 1, left, i + 1, right)
            stack.append(j)

    return best_rectangle
   
def process_foreground(item):
    name = item.name
    mask = iio.imread(item)
    img = iio.imread(img_path / name)

    for class_id, class_name in enumerate(class_names):
        if class_name == 'void':
            continue
        current_mask = (mask == class_id).astype(np.uint8)
        if class_name in crop_species:
            save_cropped_image(img, mask, current_mask, name, class_name, synthetic_fore_crop, class_id)
        elif class_name in weed_species:
            save_cropped_image(img, mask, current_mask, name, class_name, synthetic_fore_weed, class_id)

def process_background(item):
    name = item.name

    mask = iio.imread(item)
    img = iio.imread(img_path / name)

    binary_mask = (mask == 1).astype(np.uint8)
    best_rectangle = maximal_rectangle(binary_mask)

    (top, left, bottom, right) = best_rectangle
    # print(f'{synthetic_back_soil}/{name}')
    iio.imwrite(f'{synthetic_back_soil}/{name}', img[top:bottom, left:right, :])
    
if __name__ == "__main__":
    
    
    create_directories()

    img_path = Path(real_train_images)
    anno_path = Path(real_train_annotations)

    print("Cropping Individual crop and weed images...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(process_foreground, anno_path.glob('*.png')))

    print("Individual crop and weed images have been cropped and saved.")
    
    print("Cropping soil images using maximal rectangel area method...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(process_background, anno_path.glob('*.png')))
    print("soil images have been cropped and saved.")