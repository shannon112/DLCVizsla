import torch.utils.data as data
from PIL import Image
import os

class GetLoader(data.Dataset):
    def __init__(self, img_root,label_path, transform=None):
        self.img_root = img_root
        self.transform = transform

        f = open(label_path, 'r')
        data_list = f.readlines()[1:]
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            img_total_path = os.path.join(self.img_root, data[:-3])
            self.img_paths.append(img_total_path) #73252.png,2\n
            self.img_labels.append(int(data[-2])) #73252.png,2\n

    def __getitem__(self, item):
        """ Get a sample from the dataset """
        img_path, label = self.img_paths[item], self.img_labels[item]

        """ INPUT: image part """
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        """ INPUT: label part """
        return img, label

    def __len__(self):
        return self.n_data
