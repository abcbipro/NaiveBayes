###########################################################
####################### Description #######################
###########################################################

### Dependency for this project. Can be install via conda or pip
#	numpy
#	pandas
#	nltk
#	sklearn
#	pyprind

###################### Setting for read_file.py ######################

# Labels name and encode key
labels = {'ham': 0, 'spam': 1}

# Base path to read data from - default is data folder in the
# same directory with read_file.py
base_path = './data'

# Spam and ham email file name pattern 
ham_string = r'ham\.txt'
spam_string = r'spam\.txt'

# Path to save file after preprocessing
save_path = './enron.csv'


###################### Setting for read_file.py ######################

# Path to nltk package data (path when running cmd nltk.download()) 
nltk_data_path = 'F:\\nltk_data'

# Percentage of train/test data from corpus
# train + test = corpus = 10
train_percent = 7

# Max number of vocab when vectorized data
max_features = 6000