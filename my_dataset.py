import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class TranslDataset2D(Dataset):
    def __init__(self, image_dir, labels_dir, transform=None):
        """
        Args:
            image_dir (str): Directory with all the TIFF images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.tiff') or f.endswith('.tif')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, label_path = os.path.join(self.image_dir, self.image_files[idx]), os.path.join(self.labels_dir, self.image_files[idx])
        image, label = Image.open(img_path), Image.open(label_path)
        image, label = np.array(image), np.array(label)

        # adding channel dimension
        image, label = image[np.newaxis, :, : ], label[np.newaxis, :, :]

        if self.transform:
            image = self.transform(image)

        return image, label

# Example usage:
# from torchvision import transforms
# transform = transforms.Compose([transforms.ToTensor()])
# dataset = TiffDataset(image_dir='/path/to/tiff/images', transform=transform)
image_dir = '/mnt/data/mij17663/nac2ac_sep/2d_slices/NAC_PET_Tr'
labels_dir = '/mnt/data/mij17663/nac2ac_sep/2d_slices/AC_PET_Tr'

dataset = TranslDataset2D(image_dir, labels_dir)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)


img, label = next(iter(dataloader))
