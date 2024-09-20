import pandas as pd

from tqdm import tqdm

path = 'path-to-data-repository'

df_topics = pd.read_parquet(f'{path}topics03-05.parquet')

for _, row in tqdm(df_topics.iterrows()):
    with open(f'{path}old_txts/{row["file_name"]}.txt', 'a') as f:
        f.write(row["text"])