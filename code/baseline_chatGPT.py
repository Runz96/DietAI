from chagApp_openai import Vision
import pandas as pd
import time

# Load nutrition data
df_nutrition = pd.read_excel(
    '../2019-2020 FNDDS At A Glance - FNDDS Nutrient Values.xlsx', 
    sheet_name='FNDDS Nutrient Values', 
    skiprows=1
)
col_names_nutrition = df_nutrition.columns[4:]

# Define nutrient categories
dict_nutrition = {
    'basic nutrients': ['Calorie (kcal)', 'Protein (g)', 'Carbohydrate (g)', 'Total Fat (g)', 'Water (g)'],
    'sugars & fiber': ['Sugars, total (g)', 'Fiber, total dietary (g)'],
    'fatty acids': ['Fatty acids, total saturated (g)', 'Fatty acids, total monounsaturated (g)', 'Fatty acids, total polyunsaturated (g)', 'Cholesterol (mg)', 'Individual fatty acids (g)'],
    'vitamins': ['Vitamin A, RAE (mcg_RAE)', 'Vitamin C (mg)', 'Thiamin (mg)', 'Riboflavin (mg)', 'Niacin (mg)', 'Vitamin B-6 (mg)', 'Folate, total (mcg)', 'Vitamin B-12 (mcg)', 'Vitamin E (alpha-tocopherol) (mg)', 'Vitamin D (D2 + D3) (mcg)', 'Vitamin B-12, added (mcg)', 'Vitamin E, added (mg)', 'Retinol (mcg)', 'Carotene, alpha (mcg)', 'Carotene, beta (mcg)', 'Cryptoxanthin, beta (mcg)', 'Lycopene (mcg)', 'Lutein + zeaxanthin (mcg)', 'Folic acid (mcg)', 'Folate, food (mcg)', 'Folate, DFE (mcg_DFE)'],
    'minerals': ['Calcium (mg)', 'Iron (mg)', 'Magnesium (mg)', 'Phosphorus (mg)', 'Potassium (mg)', 'Sodium (mg)', 'Zinc (mg)', 'Copper (mg)', 'Selenium (mcg)', 'Choline, total (mg)', 'Vitamin K (phylloquinone) (mcg)'],
    'others': ['Caffeine (mg)', 'Theobromine (mg)', 'Alcohol (g)']
}

def create_prompt(nutrient):
    """
    Create a prompt for the Vision API to estimate nutrient quantity.
    """
    return f"""
    Could you offer a rough estimate of the quantity of the nutrient: {nutrient} in the food shown in the image, even though it lacks clear portion sizes and specific ingredients, which you might infer from the visual? Please respond only in the format "{nutrient}: amount". If you are unable to provide the estimation of nutrient quantity, please only respond with "I can't help to analyze this image." and provide the reason on a new line.
    """

def process_images(df_results, dict_nutrition, output_path):
    """
    Process images to estimate nutrient quantities using the Vision API.
    """
    df_results_drop = df_results.dropna(subset=['Link'])
    ls_indices = list(df_results_drop.index)

    for i in ls_indices:
        print("-" * 150)
        print("Index: ", i)
        url_str = df_results_drop.loc[i, 'Link']
        print("Image link: ", url_str)
        
        for categ, nutrients in dict_nutrition.items():
            for nutrient in nutrients:
                col_name_amount = nutrient + ' (GPTAmount)'
                col_name_description = nutrient + ' (GPTDescription)'
                
                if col_name_amount not in df_results_drop.columns:
                    df_results_drop[col_name_amount] = pd.NA
                if col_name_description not in df_results_drop.columns:
                    df_results_drop[col_name_description] = pd.NA
                
                if not pd.isna(df_results_drop.loc[i, col_name_amount]):
                    continue
                
                try:
                    llm = Vision("gpt-4-turbo")
                    prompt = create_prompt(nutrient)
                    response = llm.chat(prompt, url_str)
                    response_split = response.split('\n')
                    
                    if "I can't help to analyze this image." in response:                    
                        df_results_drop.loc[i, col_name_amount] = "I can't help to analyze this image."
                        df_results_drop.loc[i, col_name_description] = response_split[-1]
                    else:
                        print('🎯' * 20)
                        df_results_drop.loc[i, col_name_amount] = response
                    
                    df_results_drop.to_csv(output_path, index=False)                
                    time.sleep(3)
                except Exception as e:
                    print(e)
                    continue

if __name__ == "__main__":
    # Load results data
    df_results = pd.read_csv('')
    
    # Output path for updated results
    output_path = ''
    
    # Process images and estimate nutrients
    process_images(df_results, dict_nutrition, output_path)
