import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re #for regular expression , u could use "the","for",etc directly
import nltk #The Natural Language Toolkit (NLTK) is a platform used for building Python programs that work with human language data for applying in statistical natural language processing (NLP). 
from nltk.corpus import stopwords# stopwords are unwanted words or commonly used words 
from nltk.stem.porter import PorterStemmer #for stemming (removing 'ed','ing' from drive)
ps = PorterStemmer()

dataset = pd.read_csv('tweet_train.csv')

processed_tweet =[]

nltk.download('stopwords')# to download nltk

def my_func():
  for i in range(0, 31962):
    tweet = re.sub('[^a-zA-Z]', ' ', dataset['tweet'][i])#replace pattern  in 'tweet' in dataset with space, ^a-zA-Z] means any character that IS NOT a-z OR A-Z
    tweet = tweet.lower()
    tweet = tweet.split() #Split a string into a list where each word is a list item
    #set(stopwords.words('english'))
    tweet = [ps.stem(token) for token in tweet if not token in set(stopwords.words('english'))]    #ps.stem(token) for stemming
    tweet = ' '.join(tweet) #convert list back to string
    processed_tweet.append(tweet)

np.vectorize(my_func)() #The purpose of np.vectorize is to transform functions which are not numpy-aware (e.g. take floats as input and return floats as output) into functions that can operate on (and return) numpy arrays.


from sklearn.feature_extraction.text import CountVectorizer  #convert text to word count vectors with CountVectorizer.
cv = CountVectorizer(max_features = 3000)#Algorithms take vectors of numbers as input, therefore we need to convert documents to fixed-length vectors of numbers.
X = cv.fit_transform(processed_tweet)
X = X.toarray() #The vectors returned from a call to transform() will be sparse vectors, and you can transform them back to numpy arrays 
y = dataset['label'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()
n_b.fit(X_train, y_train)
n_b.score(X_train, y_train)
n_b.score(X_test, y_test)
n_b.score(X, y)

print(cv.get_feature_names())

#check processed_tweet
