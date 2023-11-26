from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from tqdm import tqdm
import clip
import numpy as np

class RecipeDataset(Dataset):
    def __init__(self, data: pd.DataFrame, split: str, device: str, data_path: str, image_preprocessor = None, western = True, western_dict = None):
        self.split = split
        self.data = data[data.split == split]
        self.device = device
        self.prefix = data_path + "preprocessed_data/"
        self.processor = image_preprocessor

        self.all_data = []

        for idx, row in tqdm(self.data.iterrows(), total = len(self.data)):
            if western_dict is not None:
                if western_dict[row.id] != western:
                    continue
            
            d = {
                "image": image_preprocessor(Image.open("/home/ubuntu/data/preprocessed_data/" + row.image_path)).unsqueeze(0),
                "recipe": row.instructions,
                "ingredients": row.ingredients
            }
            self.all_data.append(d)
    
    def __getitem__(self, index):
        return self.all_data[index]
    
    def __len__(self):
        return len(self.all_data)