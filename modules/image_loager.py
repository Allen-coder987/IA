import os
import torch
from torchvision import transforms
import torchvision.ops as ops
from PIL import Image

class ImageLoader:
    def __init__(self, image_dir, image_size=(640, 640)):
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def load_images(self):
        image_tensors = []
        image_names = []
        for img_name in os.listdir(self.image_dir):
            img_path = os.path.join(self.image_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
            image_tensors.append(image_tensor)
            image_names.append(img_name)
        return torch.stack(image_tensors), image_names
