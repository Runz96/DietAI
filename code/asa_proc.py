import pandas as pd
from collections import Counter

# File paths
paths = {
    "2011_2012": '../NHANES/2011-2012/DR1IFF_G.XPT',
    "2013_2014": '../NHANES/2013-2014/DR1IFF_H.XPT',
    "2015_2016": '../NHANES/2015-2016/DR1IFF_I.XPT',
    "2017_pre_pandemic": '../NHANES/2017 March-2020 Pre-pandemic/P_DR1IFF.XPT',
    "food_descriptions": '../FNDDS/2019-2020 FNDDS - Foods and Beverages.csv',
    "output": '../NHANES/top_1000_frequent_select_foods.csv'
}

# Load data
def load_data(paths):
    data = {}
    for key, path in paths.items():
        if path.endswith('.XPT'):
            data[key] = pd.read_sas(path)
        elif path.endswith('.csv'):
            data[key] = pd.read_csv(path)
    return data

# Process food frequency
def get_food_freq(df):
    df = df[df['DR1DRSTZ'] == 1]
    df['DR1IFDCD'] = df['DR1IFDCD'].astype(int)
    frequency_dict = df['DR1IFDCD'].value_counts().to_dict()
    order_dict = {key: rank for rank, (key, value) in enumerate(sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True), start=1)}
    return frequency_dict, order_dict

# Merge frequency dictionaries
def merge_frequencies(freq_dicts):
    counters = [Counter(freq) for freq in freq_dicts]
    merged_counter = sum(counters, Counter())
    return sorted(merged_counter.items(), key=lambda x: x[1], reverse=True)

# Create report
def create_report(dict_top_1000, order_dicts, food_descriptions):
    reports = []
    for idx, (food_code, _) in enumerate(dict_top_1000.items()):
        food_des = food_descriptions.get(food_code, None)
        ranks = [order_dict.get(food_code, None) for order_dict in order_dicts]
        reports.append([food_code, food_des, idx] + ranks)
    return pd.DataFrame(reports, columns=['Food code', 'Main Food description', 'Rank', 'Rank 2011-2012', 'Rank 2013-2014', 'Rank 2015-2016', 'Rank 2017 Pre-pandemic'])

# Main function
def main(paths):
    data = load_data(paths)
    order_dicts = []
    freq_dicts = []
    for key in ["2011_2012", "2013_2014", "2015_2016", "2017_pre_pandemic"]:
        freq_dict, order_dict = get_food_freq(data[key])
        freq_dicts.append(freq_dict)
        order_dicts.append(order_dict)
    
    merged_counter_sort = merge_frequencies(freq_dicts)
    dict_top_1000 = dict(merged_counter_sort[:1000])
    
    food_descriptions = data["food_descriptions"].set_index('Food code')['Main Food description'].to_dict()
    
    df_reports = create_report(dict_top_1000, order_dicts, food_descriptions)
    df_reports.to_csv(paths["output"], index=False)
    print(df_reports.head())

if __name__ == "__main__":
    main(paths)
