import pandas as pd
import numpy as np
import re

def parse_food_items(food_string):
    """
    Parses a string containing food items and their weights.
    """
    food_string = food_string.split('\n\n')[0]
    if food_string == "I can't help to analyze this text.":
        return np.nan
    lines = food_string.split('\n')
    
    food_dict = {}
    for line in lines:
        try:
            name, weight = line.split(':')
            match = re.search(r'\d+', weight)
            if 'none' in weight.lower():
                food_dict[name.strip()] = np.nan
            elif match:
                food_dict[name.strip()] = float(match.group(0))
            else:
                print(f"Error in parsing weight: {weight}")
        except ValueError as e:
            print(f"Error processing line: {line}, {e}")

    return food_dict

def match_food_codes_na(food_string):
    """
    Matches food items to None weights.
    """
    food_codes = {}
    items = food_string.split('; ')
    for item in items:
        try:
            name, codes = item.split(': ')
            food_codes[name.strip().lower()] = codes.split(', ')
        except ValueError as e:
            print(f"Error processing item: {item}, {e}")
    
    food_weights = {food: np.nan for food in food_codes.keys()}
    return food_weights

def match_food_codes(food_string, weight_dict):
    """
    Matches food items to their respective weights
    """
    normalized_weights = {key.lower(): value for key, value in weight_dict.items()}
    
    food_codes = {}
    items = food_string.split('; ')
    for item in items:
        try:
            name, codes = item.split(': ')
            food_codes[name.strip().lower()] = codes.split(', ')
        except ValueError as e:
            print(f"Error processing item: {item}, {e}")
    
    food_weights = {food: (codes, normalized_weights.get(food, np.nan)) for food, codes in food_codes.items()}
    return food_weights

def calculate_dish_nutrition(food_weights, nutrition_df, col_names_nutrition):
    """
    Calculates the total nutrition for a dish based on food weights and nutrition values.
    """
    nutrition_totals = pd.DataFrame(columns=col_names_nutrition)
    food_code_error = 0
    weight_error = 0
    weight_total = 0

    for food, (codes, weight) in food_weights.items():
        try:
            codes_float = [float(code) for code in codes if code.isdigit()]
            food_nutrition = nutrition_df[nutrition_df['Food code'].isin(codes_float)][col_names_nutrition]
            
            if pd.isna(weight):
                weight_error += 1
                print(f"Weight is None for {food}")
                continue
            
            weight_total += weight
            if not food_nutrition.empty:
                mean_nutrition = food_nutrition.mean()
                nutrition_for_food = mean_nutrition * (weight / 100)
                nutrition_totals = nutrition_totals.add(pd.DataFrame([nutrition_for_food]), fill_value=0)
        except ValueError as e:
            print(f"Error processing food item: {food}, {e}")
            food_code_error += 1
    
    return nutrition_totals, food_code_error, weight_error, weight_total

def process_results(df_results, df_nutrition, col_names_nutrition):
    """
    Processes the results DataFrame to calculate nutrition values for each dish.
    """
    ls_food_code_err = []
    ls_weight_err = []
    ls_ingredient_num = []
    ls_weight_est = []

    for i, row in df_results.iterrows():
        food_string = row['GPTFoodCode']
        weight_dict = row['GPTAmountWeight']
        
        if pd.isna(weight_dict):
            food_weights = match_food_codes_na(food_string)
            ls_food_code_err.append(0)
            ls_weight_err.append(len(food_weights))
            ls_ingredient_num.append(len(food_weights))
            ls_weight_est.append(np.nan)
        else:
            food_weights = match_food_codes(food_string, weight_dict)
            nutrition_dish, food_code_error_dish, weight_error_dish, weight_dish = calculate_dish_nutrition(food_weights, df_nutrition, col_names_nutrition)
            ls_food_code_err.append(food_code_error_dish)
            ls_weight_err.append(weight_error_dish)
            ls_ingredient_num.append(len(food_weights))
            ls_weight_est.append(weight_dish)
            
            if not nutrition_dish.empty:
                df_results.loc[i, col_names_nutrition] = nutrition_dish.values[0]

    df_results['FoodCodeError'] = ls_food_code_err
    df_results['WeightError'] = ls_weight_err
    df_results['IngredientNum'] = ls_ingredient_num
    df_results['GPTWeight'] = ls_weight_est

    df_results.to_csv('../dish_metadata.csv', index=False)

if __name__ == "__main__":
    # Load data
    df_results = pd.read_csv('../dish_metadata.csv')
    df_nutrition = pd.read_excel('../FNDDS/2019-2020 FNDDS At A Glance - FNDDS Nutrient Values.xlsx', sheet_name='FNDDS Nutrient Values', skiprows=1)
    
    # Parse food items
    df_results['GPTAmountWeight'] = df_results['GPTAmount'].apply(parse_food_items)

    # Initialize columns for nutrition values
    col_names_nutrition = df_nutrition.columns[4:]
    df_results[col_names_nutrition] = pd.DataFrame([[np.nan]*len(col_names_nutrition)], index=df_results.index)

    # Process results
    process_results(df_results, df_nutrition, col_names_nutrition)
