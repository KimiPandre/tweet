# Twitter Sentiment Analysis
Sentiment Analysis is a technique used in text mining. This dataset contains 31,962 tweets extracted using the twitter api . The tweets have been annotated  and they can be used to detect sentiment. As a part of Natural Language Processing, algorithms like SVM, Naive Bayes is used in predicting the polarity of the sentence.

<img width="522" alt="tweet" src="https://user-images.githubusercontent.com/55571522/65373129-cfbd3700-dc96-11e9-9136-51a8b596d10f.png">


# Fine Grained Sentiment Analysis
Sometimes you may be also interested in being more precise about the level of polarity of the opinion, so instead of just talking about positive, neutral, or negative opinions you could consider the following categories:

* Very positive
* Positive
* Neutral
* Negative
* Very negative

# Classification Algorithms
The classification step usually involves a statistical model like Naïve Bayes, Linear Regression, Support Vector Machines, or Neural Networks:

**Naïve Bayes** : a family of probabilistic algorithms that uses Bayes’s Theorem to predict the category of a text.

**Linear Regression** : a very well-known algorithm in statistics used to predict some value (Y) given a set of features (X).

**Support Vector Machine** : a non-probabilistic model which uses a representation of text examples as points in a multidimensional space. These examples are mapped so that the examples of the different categories (sentiments) belong to distinct regions of that space.. Then, new texts are mapped onto that same space and predicted to belong to a category based on which region they fall into.


# How to perform Twitter Sentiment Analysis:
# Tweepy: 
Tweepy, the Python client for the official Twitter API supports accessing Twitter via Basic Authentication. Tweepy gives access to       the well documented Twitter API. Tweepy makes it possible to get an object and use any method that the official Twitter API offers. 
# Tokenization:
Tokenization is the act of breaking up a sequence of strings into pieces such as words, keywords, phrases, symbols and other         elements called tokens. Tokens can be individual words, phrases or even whole sentences. In the process of tokenization, some characters  like punctuation marks are discarded. The tokens become the input for another process like parsing and text mining.

# Sentiment Analysis Metrics and Evaluation 
There are many ways in which you can obtain performance metrics for evaluating a classifier and to understand how accurate a sentiment analysis model is. One of the most frequently used is known as **cross-validation**. 
What cross-validation does is splitting the training data into a certain number of training folds (with 75% of the training data) and a the same number of testing folds (with 25% of the training data), use the training folds to train the classifier, and test it against the testing folds to obtain performance metrics (see below). The process is repeated multiple times and an average for each of the metrics is calculated.

# Precision, Recall and Accuracy
Precision, recall, and accuracy are standard metrics used to evaluate the performance of a classifier.

**Precision** measures how many texts were predicted correctly as belonging to a given category out of all of the texts that were predicted (correctly and incorrectly) as belonging to the category.

**Recall** measures how many texts were predicted correctly as belonging to a given category out of all the texts that should have been predicted as belonging to the category. We also know that the more data we feed our classifiers with, the better recall will be.

**Accuracy** measures how many texts were predicted correctly (both as belonging to a category and not belonging to the category) out of all of the texts in the corpus.
    
