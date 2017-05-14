from __future__ import division 
import numpy as np 
import pandas as pd 
import pprint
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from setting import nltk_data_path, save_path, train_percent, max_features
from datetime import datetime

def preprocess(nltk_data_path, save_path, tf_or_tfidf="tf", train_percent=7, val_percent=1, test_percent=2, max_features=1000):
	"""
	Function is use to create train, test and validation set from a csv source file
	
	The in put file format is
	///////////////////////////
	//	Email	//	Label	//	
	//	Content	//			//
	///////////////////////////
	
	Params: (almost all of this param can be found in the setting file)
	@param nltk_data_path is the path to the data of nltk lib
	@param save_path is the path where the source csv we want to read
	@param tf_or_tfidf is to choose do we want to use tf as the vectorizer or tfidf as the vectorizer
	@param train_percent, val_percent, test_percent is the percent of each set we want to create
	@param max_features is our size of the vocab

	@return 6 numpy arrays for train, test, validation set with content (X) and label (Y)
	"""
	nltk.data.path.append(nltk_data_path)
	df = pd.read_csv(save_path)
	d_size, l_size = df.shape

	print "Loading data..."
	corpus = df.loc[:, 'content'].values
	labels = df.loc[:, 'label'].values

	print "Vectorizing corpus..."
	eng_stopwords = stopwords.words('english')
	if tf_or_tfidf == "tfidf":
		vectorizer = TfidfVectorizer(min_df=1, stop_words = eng_stopwords, max_features=max_features)
	else:
		vectorizer = CountVectorizer(min_df=1, stop_words = eng_stopwords, max_features=max_features)
	corpus_vetorized = vectorizer.fit_transform(corpus)
	corpus_vetorized = np.array(corpus_vetorized.toarray())

	training_size = int(d_size/10*train_percent)
	val_size = int(d_size/10*val_percent)
	test_size = int(d_size - training_size - val_size)
	print "Training size:", training_size, "Test size:", test_size, "Validation size:", val_size
	X_train  = corpus_vetorized[:training_size]
	Y_train = labels[:training_size]
	X_val = corpus_vetorized[training_size:training_size+val_size]
	Y_val = labels[training_size:training_size+val_size]
	X_test = corpus_vetorized[training_size+val_size:]
	Y_test = labels[training_size+val_size:]
	return X_train, X_test, Y_train, Y_test, X_val, Y_val

def train(X_train, X_val, Y_train, Y_val, max_features):
	"""
	Training function, take 4 set of data and run the Multinomial NB on those data
	and after train the model, it will run the test to see what is the accuracy and 
	the F1-score of the model.  
	"""
	print "Training..."
	clf = MultinomialNB().fit(X_train, Y_train)

	print "Testing..."
	predicted = clf.predict(X_val)
	np.savetxt('result_int.csv', predicted, delimiter=',')
	print "Accuracy:", np.mean(predicted == Y_val)

	prob_predict = clf.predict_proba(X_val)
	np.savetxt('result_prob.csv', prob_predict, delimiter=',')
	penalty_predict = []
	for predict in prob_predict:
		if predict[1] > 9*predict[0]:
			penalty_predict.append(1)
		else:
			penalty_predict.append(0)
	np.savetxt('result_penalty.csv', penalty_predict, delimiter=',')
	print "Accuracy:", np.mean(penalty_predict == Y_val)

	with open('log.txt', 'a') as file:
		file.write("Date: " + str(datetime.now()) + "\n")
		file.write("Number of features: " + str(max_features) + "\n")
		text = metrics.classification_report(Y_val, predicted, target_names=["ham", "spam"])
		file.write(text)
		print(text)
		text = metrics.classification_report(Y_val, penalty_predict, target_names=["ham", "spam"])
		file.write(text)
		print(text)
		file.write("\n")

def val(tf_or_idf):
	"""
	The function run the validation phase wiht the size of vocab run from 100 to 10000.
	"""
	for i in range(100, 1000, 100):
		X_train, X_test, Y_train, Y_test, X_val, Y_val = preprocess(nltk_data_path, save_path, tf_or_tfidf=tf_or_idf, max_features=i)
		train(X_train, X_val, Y_train, Y_val, i)
	for i in range(1000, 11000, 1000):
		X_train, X_test, Y_train, Y_test, X_val, Y_val = preprocess(nltk_data_path, save_path, tf_or_tfidf=tf_or_idf, max_features=i)
		train(X_train, X_val, Y_train, Y_val, i)

def test(tf_or_idf, max_features):
	"""
	The function run on test set after we finish with validation and found the optimal size
	of the vocab.
	"""
	X_train, X_test, Y_train, Y_test, X_val, Y_val = preprocess(nltk_data_path, save_path, tf_or_tfidf=tf_or_idf, max_features=max_features)
	train(X_train, X_test, Y_train, Y_test, max_features)

if __name__ == '__main__':
	val("tf")
	val("tfidf")
	max_features = (int) raw_input("Enter number of features:")
	if_or_idf = input("Choose tf or tfidf?")
	test(tf_or_idf, max_features)