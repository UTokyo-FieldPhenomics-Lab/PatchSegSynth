import os
import glob
import numpy as np
import imageio.v3 as iio
from torch.utils.data import Dataset

class WE3DSDataset(Dataset):
    def __init__(self, images_dir, segmentations_dir, transform=None, train=False, images_dir2=None, segmentations_dir2=None):
        self.images_dir = images_dir
        self.segmentations_dir = segmentations_dir
        self.transform = transform
        self.train = train
        
        self.img_list = sorted(glob.glob(os.path.join(images_dir, '*.png')))
        self.seg_list = sorted(glob.glob(os.path.join(segmentations_dir, '*.png')))
        self.nSamples = len(self.img_list)

        if images_dir2 != None:
            img_list2 = sorted(glob.glob(os.path.join(images_dir2, '*.png')))
            seg_list2 = sorted(glob.glob(os.path.join(segmentations_dir2, '*.png')))
            self.img_list += img_list2
            self.seg_list += seg_list2
            self.nSamples = len(self.img_list)


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < len(self), 'index range error'

        img_path = self.img_list[index]
        seg_path = self.seg_list[index]
        
        img, mask = self.load_data(img_path, seg_path)
        h, w, _ = img.shape
        target = {
            'name': img_path.split('/')[-1],
            'height': h,
            'width': w
        }

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        return img, mask.long(), target

    
    def load_data(self, img_path, seg_path):
        img = iio.imread(img_path)
        mask = iio.imread(seg_path, mode='L')
        # mask = mask.astype(np.uint8)  # Ensure mask is in [0, 255]
        return img, mask - 1
