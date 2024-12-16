import os
from typing import Tuple, List

import tqdm
from PIL import Image
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import ViTImageProcessor


def process_dish_metadata(dish_metadata_paths: List[str]) -> pd.DataFrame:
    columns = ['dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein',]
    data = []
    for d_m_path in dish_metadata_paths:
        with open(d_m_path) as fopen:
            for line in fopen:
                line_list = line.strip().split(',')
                data.append(line_list[:len(columns)])
    df = pd.DataFrame.from_records(data, columns=columns)
    return df


class Nutrition5kDataset(Dataset):
    def __init__(self, 
                 dish_id_txt_path: str,
                 image_dir: str = './datasets/Nutrition5k_dataset/imagery/realsense_overhead',
                 dish_metadata_paths: List[str] = ['datasets/Nutrition5k_dataset/metadata/dish_metadata_cafe1.csv', 'datasets/Nutrition5k_dataset/metadata/dish_metadata_cafe2.csv'],
                 transform_fn: callable = None):
        dish_ids = pd.read_csv(dish_id_txt_path, header=None)[0].to_list()
        #dish_ids = dish_ids[:500]  # TODO: debug
        self.metadata = process_dish_metadata(dish_metadata_paths)
        self.metadata = self.metadata.set_index('dish_id')
        self.image_tensors = []
        self.dish_ids = []   # store existing dish_ids
        for dish_id in tqdm.tqdm(dish_ids):
            image_path = os.path.join(image_dir, f'{dish_id}/rgb.png')
            if os.path.exists(image_path):
                self.dish_ids.append(dish_id)
                with Image.open(image_path) as image:
                    processed_image = transform_fn(image, return_tensors="pt")
                    self.image_tensors.append(processed_image)
        print(f'Nutirtion5k transformed {len(self.image_tensors)} images')

    def __len__(self):
        return len(self.dish_ids)

    def __getitem__(self, idx):
        #return self.image_tensors[idx], self.metadata.loc[self.dish_ids[idx]]
        metadata = self.metadata.loc[self.dish_ids[idx]]
        item = {
            'dish_id': self.dish_ids[idx],
            'pixel_values': self.image_tensors[idx]['pixel_values'],
            'total_calories': metadata['total_calories'],
            'total_mass': metadata['total_mass'],
            'total_fat': metadata['total_fat'],
            'total_carb': metadata['total_carb'],
            'total_protein': metadata['total_protein'],
        }
        return item


class ASA24Dataset(Dataset):
    def __init__(self, 
                 csv_path: str,
                 image_dir: str = 'datasets/ASA/images',
                 transform_fn: callable = None):
        df = pd.read_csv(csv_path)
        self.file_names = df['FileName'].to_list()
        self.image_tensors = []
        for file_name in tqdm.tqdm(self.file_names):
            image_path = os.path.join(image_dir, file_name)
            with Image.open(image_path).convert('RGB') as image:
                processed_image = transform_fn(image, return_tensors="pt")
                self.image_tensors.append(processed_image)
        print(f'ASA24D transformed {len(self.image_tensors)} images')
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        item = {
            'pixel_values': self.image_tensors[idx]['pixel_values'],
            'file_name': self.file_names[idx],
        }
        return item


if __name__ == '__main__':
    # split=test
    from transformers import AutoImageProcessor
    model_name = '/HF_models/vit-base-patch16-224-in21k'
    vit_processor = AutoImageProcessor.from_pretrained(model_name)
    dish_id_txt_path_trn = './datasets/Nutrition5k_dataset/dish_ids/splits/rgb_train_ids.txt'
    dish_id_txt_path_tst = './datasets/Nutrition5k_dataset/dish_ids/splits/rgb_test_ids.txt'
    #print('train')
    #train_set = Nutrition5kDataset(dish_id_txt_path=dish_id_txt_path_trn, transform_fn=vit_processor)
    print('test')
    test_set = Nutrition5kDataset(dish_id_txt_path=dish_id_txt_path_tst, transform_fn=vit_processor)
    print(test_set[27])