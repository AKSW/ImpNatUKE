import glob
import os
import pandas as pd

from bs4 import BeautifulSoup
from tqdm import tqdm

path = 'path-to-data-repository'

df = pd.read_csv(f'{path}flat-data.csv')

all_files = glob.glob(os.path.join(f"{path}processed_pdfs/", "*.xml"))
all_files_df = pd.DataFrame([file.replace(f"{path}processed_pdfs/", '') for file in all_files], columns=['file_name'])
all_files_df['doi'] = all_files_df['file_name'].str.replace('@', '/')
all_files_df['doi'] = all_files_df['doi'].str.replace('.tei.xml','')
processed_df = df.join(all_files_df.set_index('doi'), on='doi', how='inner').reset_index(drop=True)
processed_df['file_name'] = processed_df['file_name'].str.replace('.tei.xml', '')
processed_df.to_parquet(f'{path}file-name_query14-08.parquet')

for file_name in tqdm(processed_df['file_name'].unique()):
    with open(f'{path}processed_pdfs/{file_name}.tei.xml', 'r') as f:
        data = f.read()

    bs_data = BeautifulSoup(data, "xml")

    b_title = bs_data.find('title').get_text()
    b_abstract = bs_data.find('abstract').get_text()
    b_body = bs_data.find('body').get_text()

    with open(f'{path}processed_txts/{file_name}.txt', 'a') as f:
        f.write(b_title + b_abstract + b_body)