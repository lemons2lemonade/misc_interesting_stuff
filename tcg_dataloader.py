import os
from PIL import Image
from collections import Counter
import json
from torch.utils.data import Dataset
from torchvision import transforms
import warnings

# Specify transformations
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class TripletDataset(Dataset):
    def __init__(self, jsonl_file, base_path,transform=None):
        """
        Args:
        - jsonl_file (str): Path to the JSONL file with labeled triplets.
        - base_path (str): Base path where the images are stored.
        - transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.base_path = base_path
        self.triplets = self._load_jsonl(jsonl_file)
        self.transform = transform

    def _load_jsonl(self, jsonl_file):
        data = []
        with open(jsonl_file, 'r') as file:
            for line in file:
                line_data = json.loads(line)
                data.extend([(line_data['label'], triplet) for triplet in line_data['triplets']])
        return data


    def __len__(self):
        return len(self.triplets)


    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        label = triplet[0]
        anchor = self.load_image_safely(triplet[1]['anchor'])
        positive = self.load_image_safely(triplet[1]['positive'])
        negative = self.load_image_safely(triplet[1]['negative'])

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)


        return label, anchor, positive, negative
    
    
    def load_image_safely(self, img_file):
        base_path = 'cropped_images/'
        """Attempt to load an image from base_path and img_file. Return None if unsuccessful."""
        # Check if inputs are of correct type
        if not isinstance(base_path, (str, bytes, os.PathLike)) or not isinstance(img_file, (str, bytes, os.PathLike)):
            raise TypeError("base_path and img_file must be str, bytes, or os.PathLike")

        img_file_parts = img_file.split(" ;")
        card_set, card_name = img_file_parts[1], img_file_parts[0]

        image_files = os.listdir(base_path)
        criteria = [card_set, card_name]
        match = next((f for f in image_files if all(c in f.lower() for c in criteria)), None)


        # Try to construct the full file path (in case file does not exist)
        try:
            full_path = os.path.join(base_path, match)
        except:
            print(f"Path does not exist: {match}")
            return None
        # Check if the file exists
        if not os.path.exists(full_path):
            print(f"File does not exist: {img_file}")
            return None

        # Try to open the image (in case img file is corrupted)
        try:
            warnings.filterwarnings("ignore", category=UserWarning, module='PIL')
            # Convert non-RGB to RGB
            image = Image.open(full_path).convert('RGB')

            tensor_image = transform(image)
            return tensor_image
        except IOError as e:
            print(f"Error loading image: {e}")
            return None