import pandas as pd
import requests

# Define file paths
PATH_SAMPLE = '../Nutrition5k/metadata/metadata_sample.csv'
PATH_DISH_1 = '../Nutrition5k/metadata/dish_metadata_cafe1.csv'
PATH_DISH_2 = '../Nutrition5k/metadata/dish_metadata_cafe2.csv'
OUTPUT_PATH = '../Nutrition5k/metadata/dish_metadata_available.csv'

# Load data
df_sample = pd.read_csv(PATH_SAMPLE)
df_dish_1 = pd.read_csv(PATH_DISH_1, header=None, engine='python', on_bad_lines='skip')
df_dish_2 = pd.read_csv(PATH_DISH_2, header=None, engine='python', on_bad_lines='skip')

def process_dish_data(df_dish):
    """Process dish data to extract relevant information."""
    dish_properties = {
        'dish_id': [],
        'total_calories': [],
        'total_mass': [],
        'total_fat': [],
        'total_carbs': [],
        'total_protein': [],
        'ingredients': [],
        'sorted_ingredients': []
    }

    for _, row in df_dish.iterrows():
        ingredients = [
            row[i + 1] for i in range(1, len(row) - 1)
            if isinstance(row[i], str) and row[i].startswith("ingr_") and isinstance(row[i + 1], str)
        ]
        if ingredients:
            dish_properties['dish_id'].append(row[0])
            dish_properties['total_calories'].append(row[1])
            dish_properties['total_mass'].append(row[2])
            dish_properties['total_fat'].append(row[3])
            dish_properties['total_carbs'].append(row[4])
            dish_properties['total_protein'].append(row[5])
            sorted_ingredients = sorted(ingredients)
            dish_properties['ingredients'].append(ingredients)
            dish_properties['sorted_ingredients'].append(sorted_ingredients)

    df_ingredients = pd.DataFrame(dish_properties)
    df_ingredients['ingredients'] = df_ingredients['ingredients'].apply(', '.join)
    df_ingredients['sorted_ingredients'] = df_ingredients['sorted_ingredients'].apply(', '.join)

    return df_ingredients

def remove_duplicate_ingredients(df):
    """Remove duplicate ingredients based on sorted ingredients and total mass."""
    df_sorted = df.sort_values(by=['sorted_ingredients', 'total_mass'], ascending=[True, False])
    df_unique = df_sorted.drop_duplicates(subset='sorted_ingredients', keep='first')
    df_unique = df_unique.sort_values(by='dish_id')
    return df_unique

def filter_unavailable_images(df):
    """Filter out dishes with unavailable images."""
    indices_to_remove = []
    for i in range(len(df)):
        dish_id = df.loc[i, 'dish_id']
        url_str = f'/{dish_id}/rgb.png'
        req_url = requests.get(url_str)
        if req_url.status_code != 200:
            print(f'Image URL is not accessible for dish_id: {dish_id}')
            indices_to_remove.append(i)

    df_filtered = df.drop(indices_to_remove)
    df_filtered.reset_index(drop=True, inplace=True)
    return df_filtered

def main():
    df_ingredients_1 = process_dish_data(df_dish_1)
    df_ingredients_2 = process_dish_data(df_dish_2)
    df_ingredients = pd.concat([df_ingredients_1, df_ingredients_2], ignore_index=True)

    df_ingredients = remove_duplicate_ingredients(df_ingredients)
    df_ingredients_available = filter_unavailable_images(df_ingredients)
    df_ingredients_available.to_csv(OUTPUT_PATH, index=False)
    print("Processed data saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
