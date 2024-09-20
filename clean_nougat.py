import glob
import os
import markdown

import pandas as pd

from bs4 import BeautifulSoup
from tqdm import tqdm

path = 'path-to-data-repository'

all_files = pd.Series(glob.glob(os.path.join(f"{path}nougat_output/", "*.md")))
all_files = all_files.str.replace(f'{path}nougat_output/','')
all_files = all_files.str.replace('.md','')

for file_name in tqdm(all_files):
    with open(f"{path}nougat_output/{file_name}.md", 'r') as f:
        htmlmarkdown = markdown.markdown(f.read())

    bs_data = BeautifulSoup(htmlmarkdown, "html")

    with open(f'{path}nougat_txts/{file_name}.txt', 'a') as f:
        f.write(bs_data.get_text())