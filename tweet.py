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

from sklearn.model_selection import KFold
kf = KFold(n_splits = 10, shuffle = False, random_state = 42)

from sklearn.linear_model import LinearRegression #model should be for continoous 'y' for this dataset
lr = LinearRegression()

scores = []

for train_index,test_index in kf.split(X):
    print('Train: ',train_index)
    print('Test: ',test_index)
    
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    lr.fit(X_train,y_train)
    scores.append(lr.score(X_test, y_test))

print(np.mean(scores))


from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()
n_b.fit(X_train, y_train)
n_b.score(X_train, y_train)
n_b.score(X_test, y_test)
n_b.score(X, y)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_train, y_train)
lr.score(X_test, y_test)
lr.score(X, y)

from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X_train, y_train)
svc.score(X_train, y_train)
svc.score(X_test, y_test)
svc.score(X, y)

from sklearn.svm import SVC
svc = SVC(kernel = 'poly')
svc.fit(X_train, y_train)
svc.score(X_train, y_train)
svc.score(X_test, y_test)
svc.score(X, y)

from sklearn import metrics
m = metrics.confusion_matrix(y_test,y_pred)
print(m) 

print("Precision score: ", metrics.precision_score(y_test,y_pred))
print("Recall score: ", metrics.recall_score(y_test,y_pred))
print("Fscore: ", metrics.f1_score(y_test,y_pred))

print(cv.get_feature_names())

#check processed_tweet
