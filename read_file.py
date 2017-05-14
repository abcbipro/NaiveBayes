import pandas as pd 
import os, re
import numpy as np 
import pyprind
from setting import *  

"""
Preprocess from separate email text files to one big csv file with content and label.
The path for save the csv is in the setting file.
"""

pbar = pyprind.ProgBar(33716)
df = pd.DataFrame()
i = 0
print 'Reading file...'
for root, dirs, files in os.walk(base_path, topdown=True):
	for name in files:
		if re.search(spam_string, name):
			with open(os.path.join(root, name), 'r') as file:
				txt = ''
				for line in file:
					txt += line.decode('utf-8','ignore').encode("utf-8")
				df = df.append([[txt, labels['spam']]], ignore_index = True)
			i += 1
		if re.search(ham_string, name):
			with open(os.path.join(root, name), 'r') as file:
				txt = ''
				for line in file:
					txt += line.decode('utf-8','ignore').encode("utf-8")
				df = df.append([[txt, labels['ham']]], ignore_index = True)
			i += 1
		pbar.update()
print 'Done, readed', i, 'files'
df.columns = ['content', 'label']
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv(save_path, index=False)
print 'Saved in', save_path
