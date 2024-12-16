import pandas as pd
import os
import requests
import json

# API details
url = "https://vision.foodvisor.io/api/1.0/en/analysis/"
headers = {"Authorization": "Api-Key PMSGGxyY.P1EYSrWEl1DL2OFgmLYdcuyVaTD7fvYd"}

# Function to process images and save results
def process_and_save_image_data(row):
    image_name = row['FileName']
    path_image_file = os.path.join(path_image, image_name)
    
    with open(path_image_file, "rb") as image:
        response = requests.post(url, headers=headers, files={"image": image})
        response.raise_for_status()
        
    data = response.json()
    results_name = image_name.replace('.png', '.json')
    path_results_file = os.path.join(path_results_save, results_name)
    
    with open(path_results_file, 'w') as file:
        json.dump(data, file, indent=4)  # The indent parameter is optional, but it makes the file human-readable

if __name__ == "__main__":
    # File paths
    path_data = '../ASA24_GPT_baseline.csv'
    path_image = '../baseline/'
    path_results_save = '../ASA24_GPTFoodCodes_nutrition.csv'

    # Load data
    df_data = pd.read_csv(path_data)
    
    # Iterate through DataFrame rows
    for index, row in df_data.iterrows():
        process_and_save_image_data(row)
