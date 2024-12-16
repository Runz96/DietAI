import pandas as pd
from fractions import Fraction
from chagApp_openai import Vision
import numpy as np
import time
import inflect

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def has_common_element(list1, list2):
    return list(set(list1).intersection(set(list2)))

def extract_unit(value):
    return ' '.join(value.split(' ', 1)[1:])

def extract_amount(value):
    return value.split(' ', 1)[0]

def singularize(word, p):
    return p.singular_noun(word) or word

def create_prompt(food_description, portion_data, type='shot'):
    if type == 'shot':
        return f"""The image displays the food described as {food_description}. Please assess the portion size shown in the image and 
choose the option that most accurately represents it from the following choices: {portion_data}. Respond with only the selected portion size. 
If you cannot analyze the image, reply with "I can't help to analyze this image." and specify the reason on a new line."""
    else:
        return f"""The image displays the food described as {food_description}. Please evaluate the portion size presented and specify the quantity: 
how many {portion_data} of the food are visible in the image? Please respond only with the numerical value. 
If you cannot analyze the image, reply with "I can't help to analyze this image." and specify the reason on a new line."""

def update_dataframe(df, index, response, column_desc, column_reason):
    response_splits = response.split('\n')
    df[column_desc] = df[column_desc].astype(object)
    df[column_reason] = df[column_reason].astype(object)
    if response_splits[0] == "I can't help to analyze this image.":
        df.loc[index, column_desc] = response_splits[0]
        df.loc[index, column_reason] = response_splits[1]
    else:
        df.loc[index, column_desc] = response

def process_dataframe(df_results, df_image_link, p):
    ls_label_amount = []
    ls_label_unit = []
    ls_portion_shot = []
    for i in range(len(df_results)):
        str_portion = df_results.loc[i, 'Portion']
        str_food_code = df_results.loc[i, 'FoodCodeCommon']
        ls_label_amount.append(float(Fraction(extract_amount(str_portion))))
        ls_label_unit.append(singularize(extract_unit(str_portion), p))
        df_image_link_sel = df_image_link[df_image_link['FoodCode'] == str_food_code].copy()
        df_image_link_sel.sort_values(by=['Multiplier'], ascending=False, inplace=True)
        ls_portions = df_image_link_sel['Portion'].tolist()
        ls_portion_shot.append(' ,'.join(ls_portions))
    df_results['LabelAmount'] = ls_label_amount
    df_results['LabelUnit'] = ls_label_unit
    df_results['PortionShot'] = ls_portion_shot
    df_results['FC_Description'] = df_results['FC_Description'].replace(to_replace=r'\bNFS\b', value='not further specified subcategory assigned', regex=True)
    df_results['FC_Description'] = df_results['FC_Description'].replace(to_replace=r'\bNS\b', value='not specified subcategory', regex=True)
    return df_results

def analyze_portions(df, llm, column_desc, column_reason, portion_data, type='shot'):
    ls_url_str = df['Link'].tolist()
    for i, url_str in enumerate(ls_url_str):
        if not pd.isna(df.loc[i, 'FC_Description']):
            food_description = df.loc[i, 'FC_Description']
            try:
                prompt = create_prompt(food_description, portion_data[i], type)
                response = llm.chat(prompt, url_str)
                update_dataframe(df, i, response, column_desc, column_reason)
                df.to_csv('../ASA24_GPTFoodCodes_portion.csv')
                time.sleep(10)
            except Exception as e:
                print(e)
                continue

def main():
    # Load data
    df_results = pd.read_csv('../df_results.csv')
    df_image_link = pd.read_csv('../df_image_link.csv')
    df_results_portion = pd.read_csv('../ASA24_GPTFoodCodes_portion.csv')

    # Initialize inflect engine
    p = inflect.engine()

    # Process dataframe
    df_results = process_dataframe(df_results, df_image_link, p)
    df_results.to_csv('../output.csv', index=False)

    # Initialize Vision API
    llm = Vision("gpt-4-turbo")

    # Analyze portions
    if 'GPTPortionDescription' not in df_results_portion.columns:
        df_results_portion['GPTPortionDescription'] = [np.nan] * len(df_results_portion)
        df_results_portion['GPTPortionReason'] = [np.nan] * len(df_results_portion)
    analyze_portions(df_results_portion, llm, 'GPTPortionDescription', 'GPTPortionReason', df_results_portion['PortionShot'])

    if 'GPTPortionAmount' not in df_results_portion.columns:
        df_results_portion['GPTPortionAmount'] = [np.nan] * len(df_results_portion)
        df_results_portion['GPTPortionAmountReason'] = [np.nan] * len(df_results_portion)
    analyze_portions(df_results_portion, llm, 'GPTPortionAmount', 'GPTPortionAmountReason', df_results_portion['LabelUnit'], type='amount')

if __name__ == "__main__":
    main()
