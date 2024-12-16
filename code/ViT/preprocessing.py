import os
import urllib.request
import fire
import pandas as pd
from tqdm.auto import tqdm


def download_images_Nut5K(
    data_xlsx_path: str = '../Nutrition5K/dish_metadata_largest_nutrient_baseline.xlsx',
    out_dir: str = '../Nutrition5K/images'
):
    df = pd.read_excel(data_xlsx_path)
    for _, row in tqdm(df.iterrows()):
        file_name = row['FileName']
        url = row['Link']
        out_file = f'{out_dir}/{file_name}'
        urllib.request.urlretrieve(url, out_file)


def download_images_ASA(
    data_csv_path: str = '../ASA24_GPT_baseline.csv',
    out_dir: str = '../ASA/images'
):
    df = pd.read_csv(data_csv_path)
    for _, row in tqdm(df.iterrows()):
        file_name = row['FileName']
        url = row['Link']
        out_file = f'{out_dir}/{file_name}'
        if not os.path.exists(out_file):
            urllib.request.urlretrieve(url, out_file)


if __name__ == '__main__':
    fire.Fire()