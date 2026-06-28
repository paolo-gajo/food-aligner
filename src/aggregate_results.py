import pandas as pd
import numpy as np
import os
import re
import json

results_dir = './results/alignment/en-it/test/GZ-GOLD_301_BertTokenizerFast_en-it/mbert'
model_name = results_dir.split('/')[-1]
df = pd.DataFrame()
for root, dirs, files in os.walk(results_dir):
    dir_name = root.split('/')[-1]
    metrics_name = 'metrics.json'
    if metrics_name in files:
        json_path = os.path.join(root, metrics_name)
        with open(json_path, 'r', encoding='utf8') as f:
            data = json.load(f)
        model_dict = {
            'model_name': re.search(r'^(.*)_', dir_name).group(1),
            'f1': data["test_f1"],
            'exact': data["test_exact"],
            'p': re.search(r'_P(\d\.\d)_', dir_name).group(1),
            'langs': re.search(r'([a-z]{2}-[a-z]{2})(-[a-z]{2})?', dir_name).group(0),
        }
        df_temp = pd.DataFrame(model_dict, index=[0])
        df = pd.concat([df, df_temp])

df = df.sort_values(by=['model_name'])
df_grouped = df.groupby(['p']).agg({'exact': [np.mean, np.std], 'f1': [np.mean, np.std]})

# ['exact'].apply(lambda x: np.mean(list(x))).tolist()
# std = df.groupby(['p'])
# ['exact'].apply(lambda x: np.std(list(x))).tolist()

# print(mean)
# print(std)
df_grouped = df_grouped.round(2)
# df_grouped['f1']['mean'].apply(lambda x: "{:.2f}".format(x))
# df_grouped['f1']['std'].apply(lambda x: "{:.2f}".format(x))
# df_grouped['exact']['mean'].apply(lambda x: "{:.2f}".format(x))
# df_grouped['exact']['std'].apply(lambda x: "{:.2f}".format(x))
# df_grouped['exact'].apply(lambda x: "{:.2f}".format(x))
print(df_grouped)
df_grouped.to_csv(os.path.join(results_dir, f'{model_name}_aggregated_results.csv'), float_format='%.2f')
df_grouped.to_latex(os.path.join(results_dir, f'{model_name}_aggregated_results.tex'), float_format='%.2f')