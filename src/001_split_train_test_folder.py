import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from configuration import WE3DS_images, WE3DS_annotations

from configuration import real_train_images, real_train_annotations, real_test_images, real_test_annotations

# 创建文件夹
os.makedirs(real_train_images, exist_ok=True)
os.makedirs(real_train_annotations, exist_ok=True)
os.makedirs(real_test_images, exist_ok=True)
os.makedirs(real_test_annotations, exist_ok=True)

from configuration import WE3DS_train_txt, WE3DS_test_txt
# 读取train.txt和test.txt文件
with open(WE3DS_train_txt, 'r') as file:
    train_files = [line.strip() for line in file]

with open(WE3DS_test_txt, 'r') as file:
    test_files = [line.strip() for line in file]

# 定义复制文件的函数
def copy_files(image_id, dst_image_folder, dst_annotation_folder):
    img_file = f'img_{image_id}.png'
    src_image = os.path.join(WE3DS_images, img_file)
    dst_image = os.path.join(dst_image_folder, img_file)
    src_annotation = os.path.join(WE3DS_annotations, img_file)
    dst_annotation = os.path.join(dst_annotation_folder, img_file)
    if os.path.exists(src_image) and os.path.exists(src_annotation):
        shutil.copy(src_image, dst_image)
        shutil.copy(src_annotation, dst_annotation)

if __name__ == "__main__":
    print("Copying images from WE3DS to SYNTHETIC_PATH you set in configuration.py (default path: '../Synthetics')... this process may take several minutes")
    
    # move images to train
    with ThreadPoolExecutor() as executor:
        for image_id in train_files:
            executor.submit(copy_files, image_id, real_train_images, real_train_annotations)

    # move images to test
    with ThreadPoolExecutor() as executor:
        for image_id in test_files:
            executor.submit(copy_files, image_id, real_test_images, real_test_annotations)

    print("Images and annotations have been copied to Train and Test folders based on train.txt and test.txt.")