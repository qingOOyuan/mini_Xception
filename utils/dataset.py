import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import *

from .DataAugment import *

# Initialize height and width shift range objects
height_shift_range = HeightShiftRange(0.1)
width_shift_range = WidthShiftRange(0.1)

class FER2013Dataset(Dataset):
    def __init__(self, mode, input_size):
        super(FER2013Dataset, self).__init__()
        # Load data from CSV file corresponding to the mode (train or test)
        self.data = np.array(pd.read_csv(os.path.join("dataset", f"{mode}.csv")))
        self.input_size = input_size
        
        # If mode is 'train', apply augmentation and transformation
        if mode == "train":
            self.augmentation = Augment([
                SaltPepperNoise(0.05),
                WidthShiftRange(0.1),
                HeightShiftRange(0.1)
            ])
            
            self.transform = transforms.Compose([
                ToTensor(),
                ColorJitter(brightness=0.2),
                RandomRotation(10),
                RandomHorizontalFlip(0.5)
            ])
        else:
            # For non-training modes, use default augmentation and only convert to tensor
            self.augmentation = Augment()
            self.transform = transforms.Compose([ToTensor()])

    def __getitem__(self, index):
        # Extract label and image from dataset
        label, img_str, _ = self.data[index]
        # Convert pixel string to numpy array
        image_data = np.array([int(pixel) for pixel in img_str.split()], dtype=np.uint8)
        # Reshape image to 48x48
        image = np.reshape(image_data, (48, 48))
        # Resize image using cv2, which fills with zeros (this might cause performance drop due to padding)
        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Apply custom augmentation
        image = self.augmentation(image)
        # Apply torchvision transformations
        image = self.transform(image)
        
        return label, image

    def __len__(self):
        # Return the total number of items in the dataset
        return len(self.data)
