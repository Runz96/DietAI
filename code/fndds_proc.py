import pandas as pd
import numpy as np

# Define file paths
EXCEL_FILE_PATH = '../FNDDS/2019-2020 FNDDS At A Glance - Foods and Beverages.xlsx'
CSV_FILE_PATH = '../FNDDS/2019-2020 FNDDS - Foods and Beverages.csv'

def load_data(excel_file_path):
    """Load data from the Excel file and select relevant columns."""
    data = pd.read_excel(excel_file_path, sheet_name='Food and Beverages', header=1)
    data = data[['Food code', 'Main food description', 'Additional food description']]
    return data

def replace_abbreviations(data):
    """Replace abbreviations in the descriptions."""
    data.replace(to_replace=r'\bNFS\b', value='not further specified subcategory assigned', regex=True, inplace=True)
    data.replace(to_replace=r'\bNS\b', value='not specified subcategory', regex=True, inplace=True)
    return data

def create_food_description(data):
    """Create a comprehensive food description."""
    descriptions = []
    for i, row in data.iterrows():
        main_description = row['Main food description']
        if pd.isna(main_description):
            descriptions.append(np.nan)
        else:
            main_str_start = 'The image shows the food category of '
            main_description = main_str_start + main_description[0].lower() + main_description[1:] + '.'
            add_description = row['Additional food description']
            if not pd.isna(add_description):
                main_description += ' Additional details include ' + add_description[0].lower() + add_description[1:] + '.'
            descriptions.append(main_description)
    data['Food description'] = descriptions
    return data

def save_data(data, csv_file_path):
    """Save the cleaned data to a CSV file."""
    data.drop(columns=['Main food description', 'Additional food description'], inplace=True)
    data.dropna(inplace=True)
    data.to_csv(csv_file_path, index=False)

def main():
    data = load_data(EXCEL_FILE_PATH)
    data = replace_abbreviations(data)
    data = create_food_description(data)
    save_data(data, CSV_FILE_PATH)
    print("Data processing complete and saved to CSV.")

if __name__ == "__main__":
    main()
