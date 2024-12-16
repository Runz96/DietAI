from PIL import Image
import pandas as pd
import requests
import json
import os
import time

def prepare_image(image_path):
    """
    Resize the image to 544x544 pixels and convert it to JPEG format.
    """
    with Image.open(image_path) as img:
        img = img.resize((544, 544))
        rgb_img = img.convert('RGB')
        jpeg_path = image_path.replace('.png', '.jpeg')
        rgb_img.save(jpeg_path)
    return jpeg_path

def post_image_binary(api_key, image_path):
    """
    Post the image to the food recognition API and get the response.
    """
    url = f"https://api-2445582032290.production.gw.apicast.io/v1/foodrecognition?user_key={api_key}"
    headers = {'Content-Type': 'image/jpeg'}
    
    with open(image_path, 'rb') as img:
        data = img.read()
        
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()  # Raises an HTTPError if the response was an error
    return response.json()

def process_images(api_key, df_data, path_image, path_results_save, start=0, end=100):
    """
    Process a range of images, send them to the API, and save the results.
    """
    for i in range(start, end):
        image_name = df_data.loc[i, 'FileName']
        path_image_file = os.path.join(path_image, image_name)
        jpeg_image_path = prepare_image(path_image_file)
        
        try:
            data = post_image_binary(api_key, jpeg_image_path)
            results_name = image_name.replace('.png', '.json')
            path_results_file = os.path.join(path_results_save, results_name)
            
            with open(path_results_file, 'w') as file:
                json.dump(data, file, indent=4)  # The indent parameter makes the file human-readable
            
        except requests.exceptions.RequestException as e:
            print(f'Error in image {image_name}: {e}')
        
        time.sleep(3)  # Sleep for 3 seconds to avoid exceeding the rate limit of the API

if __name__ == "__main__":
    # File paths
    path_data = '../ASA24_GPT_baseline.csv'
    path_image = '../baseline/'
    path_results_save = '../ASA24_GPTFoodCodes_nutrition.csv'

    # Load data
    df_data = pd.read_csv(path_data)

    # API key
    api_key = ""

    # Process images
    process_images(api_key, df_data, path_image, path_results_save)
