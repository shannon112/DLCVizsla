import torch.utils.data as data
from PIL import Image
import os
import glob

class GetLoader(data.Dataset):
    def __init__(self, img_root, transform=None):
        self.img_root = img_root
        self.transform = transform
        self.img_paths = sorted(glob.glob(os.path.join(img_root, '*.png')))
        self.len = len(self.img_paths)

    def __getitem__(self, item):
        """ Get a sample from the dataset """
        img_path = self.img_paths[item]

        """ INPUT: image part """
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        """ INPUT: image_name part """
        img_fn = img_path.split('/')[-1]
        return img, img_fn

    def __len__(self):
        return self.len
