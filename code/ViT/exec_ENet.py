"""
Vit with Multi-task ElasticNet
"""
import pickle

from tqdm.auto import tqdm
import numpy as np
import fire
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ViTModel
from sklearn import linear_model

from dataset_utils import Nutrition5kDataset, ASA24Dataset


def train(
    pt_save_path: str,
    model_name: str = '/HF_models/vit-base-patch16-224-in21k', 
    dataset_name: str = 'Nutrition5k',
    batch_size: int = 64,
    seed: int = 1993   
):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    vit_processor = AutoImageProcessor.from_pretrained(model_name)
    if dataset_name == 'Nutrition5k':
        dish_id_txt_path_trn = './datasets/Nutrition5k_dataset/dish_ids/splits/rgb_train_ids.txt'
        dish_id_txt_path_tst = './datasets/Nutrition5k_dataset/dish_ids/splits/rgb_test_ids.txt'
        train_set = Nutrition5kDataset(dish_id_txt_path=dish_id_txt_path_trn, transform_fn=vit_processor)
        print(f'Loaded Nutrition5k dataset with {len(train_set)} training samples')
        #test_set = Nutrition5kDataset(dish_id_txt_path=dish_id_txt_path_tst, transform_fn=vit_processor)
    else:
        raise ValueError(f'Unknown dataset name: {dataset_name}')
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    #test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # prepare ViT Model
    vit_model = ViTModel.from_pretrained(model_name).to(device)
    vit_model.eval()

    # do a quick test
    hid_embs = []
    all_targets = [[], [], [], [], []]
    for batch in tqdm(train_dataloader, desc="ViT Embeddings"):
        batch['pixel_values'] = batch['pixel_values'].squeeze(dim=1)
        # collect targets
        batch.pop('dish_id')
        batch_calories = batch.pop('total_calories')   # list
        batch_mass = batch.pop('total_mass')
        batch_fat = batch.pop('total_fat')
        batch_carb = batch.pop('total_carb')
        batch_protein = batch.pop('total_protein')
        all_targets[0].extend(batch_calories)
        all_targets[1].extend(batch_mass)
        all_targets[2].extend(batch_fat)
        all_targets[3].extend(batch_carb)
        all_targets[4].extend(batch_protein)
        # collect embedding
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = vit_model(**batch)
            batch_hid_embs = outputs.pooler_output
            hid_embs.append(batch_hid_embs.detach().cpu().numpy())
    hid_embs = np.concatenate(hid_embs, axis=0).astype(np.float64)   # shape=(N, 768)
    all_targets = np.array(all_targets, dtype=np.float64).transpose()   # shape=(N, 5)
    print(f'After ViT: {hid_embs.shape=}, {all_targets.shape=}')
    # start to train the ElasticNet
    clf = linear_model.MultiTaskElasticNet(random_state=seed)
    clf.fit(hid_embs, all_targets)
    # save model
    with open(pt_save_path, 'wb') as fwrite:
        pickle.dump(clf, fwrite)
    print(f'Saved model to {pt_save_path}')


def test(
    dataset_path: str,
    output_path: str,
    pt_load_path: str = '/scratch/jlu229/NutritionEst-baselines/exps/ViT-ENet/MultiTaskElasticNet.pkl',
    dataset_name: str = 'ASA24',
    model_name: str = '/scratch/jlu229/HF_models/vit-base-patch16-224-in21k', 
    batch_size: int = 64,
    seed: int = 1993   
):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    vit_processor = AutoImageProcessor.from_pretrained(model_name)
    if dataset_name == 'Nutrition5k':
        test_set = Nutrition5kDataset(dish_id_txt_path=dataset_path, transform_fn=vit_processor)
        id_key = 'dish_id'
    elif dataset_name == 'ASA24':
        test_set = ASA24Dataset(csv_path=dataset_path, transform_fn=vit_processor)
        id_key = 'file_name'
    else:
        raise ValueError(f'Unknown dataset name: {dataset_name}')
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # prepare models
    vit_model = ViTModel.from_pretrained(model_name).to(device)
    vit_model.eval()
    with open(pt_load_path, 'rb') as fread:
        clf = pickle.load(fread)  # type: linear_model.MultiTaskElasticNet

    hid_embs = []
    all_file_names = []
    for ori_batch in tqdm(test_dataloader, desc="ViT Embeddings"):
        batch = {}
        batch['pixel_values'] = ori_batch['pixel_values'].squeeze(dim=1)
        batch_file_names = ori_batch.pop(id_key)   # list
        all_file_names.extend(batch_file_names)
        # collect embedding
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = vit_model(**batch)
            batch_hid_embs = outputs.pooler_output
            hid_embs.append(batch_hid_embs.detach().cpu().numpy())
    hid_embs = np.concatenate(hid_embs, axis=0).astype(np.float64)   # shape=(N, 768)
    predictions = clf.predict(hid_embs)
    print(f'{predictions.shape=}')
    # save predictions
    data_records = []
    for file_name, pred in zip(all_file_names, predictions):
        record = [file_name] + pred.tolist()
        data_records.append(record)
    out_df = pd.DataFrame.from_records(data_records, columns=['FileName', 'TotalCalories', 'TotalMass', 'TotalFat', 'TotalCarb', 'TotalProtein'])
    out_df.to_csv(output_path, index=False)
    print(f'Saved predictions {out_df.shape} to {output_path}')


if __name__ == '__main__':
    fire.Fire()