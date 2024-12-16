import pandas as pd
import numpy as np

# File paths
path_portion_code = '../FoodWeights.csv'
path_results = '../ASA24_GPTFoodCodes_portion.csv'
path_nutrition = '../FNDDS/2019-2020 FNDDS At A Glance - FNDDS Nutrient Values.xlsx'

# Load data
df_portion_code = pd.read_csv(path_portion_code)
df_results = pd.read_csv(path_results)
df_nutrition = pd.read_excel(path_nutrition, sheet_name='FNDDS Nutrient Values', skiprows=1)
col_names_nutrition = df_nutrition.columns[4:]

# Initialize columns in df_results
df_results['TotalWeight'] = np.nan
for nutrient_name in col_names_nutrition:
    df_results[nutrient_name] = np.nan

# Helper function to get portion weight
def get_portion_weight(food_code, portion_code, portion_subcode):
    df_filtered = df_portion_code[(df_portion_code['Food Code'] == food_code) & (df_portion_code['Portion Code'] == portion_code)]
    if len(df_filtered) > 1:
        df_filtered = df_filtered[df_filtered['Subcode'] == portion_subcode]
    if len(df_filtered) == 1:
        return df_filtered['Portion Weight'].values[0]
    return np.nan

# Helper function to get nutrition values
def get_nutrition_values(food_code, weight):
    nutrition_values = {}
    food_code_nutrition = df_nutrition[df_nutrition['Food code'] == food_code]
    if len(food_code_nutrition) == 1 and weight >= 0:
        for nutrition_name in col_names_nutrition:
            nutrition_values[nutrition_name] = food_code_nutrition[nutrition_name].values[0] * weight * 0.01
    else:
        print(f"Food code {food_code} not found in nutrition data")
    return nutrition_values

# Process each row in df_results
for i in range(len(df_results)):
    food_code = df_results.loc[i, 'FoodCodeCommon']
    portion_code = df_results.loc[i, 'PortionCode']
    portion_subcode = df_results.loc[i, 'PortionSubCode']
    
    weight = get_portion_weight(food_code, portion_code, portion_subcode)
    df_results.loc[i, 'TotalWeight'] = weight
    
    nutrition_values = get_nutrition_values(food_code, weight)
    for nutrient_name, value in nutrition_values.items():
        df_results.loc[i, nutrient_name] = value

# Save the updated dataframe
output_path = '../ASA24_GPTFoodCodes_nutrition.csv'
df_results.to_csv(output_path, index=False)
