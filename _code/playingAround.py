######
#The massive number of packages I'm using in this short
#code sample
import numpy as np
import pandas as pd
import sklearn as skl
import scipy.stats as sp
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.grid_search import RandomizedSearchCV
import xgboost as xgb
######

#Assuming that you've run python from within the twitter-sentiment
#folder. Otherwise you may need to add in the full path
twitter_df = pd.read_csv("_data/MaskedDataUsable.csv",
                         quotechar = '"', skipinitialspace = True)

#Pulling the relevant columns from the pandas data frame. tweets are the actual
#text of the tweets, sentiment is the target. 0 for negative, 1 for positive sentiment.
tweets = twitter_df.SentimentText
sentiment = twitter_df.Sentiment


#The most basic nlp possible. My next step (not included in this sample
#will be to work more closely with the text itself to do some feature engineering
stemmer = SnowballStemmer('english')
tweets = tweets.apply(stemmer.stem)

tfidf = TfidfVectorizer(strip_accents = "ascii", 
                        ngram_range = (1,3),
                        max_features = 2000, 
                        stop_words = 'english', 
                        max_df = .95)

#The test data that will end up being used by kaggle to judge my model is
#marked with a -1. Removing those values, then turning the remainder
#into a Document-Text Matrix
test_data = (sentiment != -1)
tweets = tweets[test_data]
sentiment = sentiment[test_data]

#The main data set I will be doing analysis on. ~1,500,000 x 2,000 sparse
#matrix
tweet_dtm = tfidf.fit_transform(tweets)

#Will do cv on the training set, but then will want the final
#testing to be on a separate test set. Make those here.
X_train, X_test, y_train, y_test = train_test_split(tweet_dtm, sentiment,
													test_size = .3333)

#These are the models I ended up deciding to put together as an ensemble method.
#Not pictured is the many hours I spent trying to balance predictive power /
#with how efficiently it could deal with the massive amount of data.
#Didn't make the cut: Kernel SVM or PC Regression (way too slow), sklearn's
#boosting and bagging algorithms (didn't perform nearly as well or as fast as
#or as well as xgboost), etc...
rf = RandomForestClassifier(n_estimators = 200)

bst = xgb.XGBClassifier(n_estimators = 100, 
						 silent = False, objective = 'binary:logistic')
						 
logit = LogisticRegression(solver = 'sag', verbose = 2)

multi_nb = MultinomialNB()

sgd = SGDClassifier(penalty = 'l2')



#This is where we do the parameter tuning. Limit it to 100 (x 3-fold cv) random
#iterations because to do an exhaustive search across the paramater space would 
#be prohibitively computationally expensive :/
est_dist_rf = {'max_depth':[5, 10, 15], 'min_weight_fraction_leaf':sp.uniform(),
			 'class_weight':[None, 'balanced', 'balanced_subsample']}
est_dist_bst = {'learning_rate':sp.uniform(), 'gamma':sp.uniform(scale = 10),
				'reg_alpha':sp.uniform(scale = 15)}
est_dist_logit = {'C':sp.uniform()}
est_dist_nb = {'alpha':sp.uniform()}
est_dist_sgd = {'alpha':sp.uniform(scale = .0002)}

#Number of iterations
n_iter = 100

#Do the search
grid_search_rf = RandomizedSearchCV(rf, est_dist_rf, n_iter)
grid_search_bst = RandomizedSearchCV(bst, est_dist_bst, n_iter)
grid_search_logit = RandomizedSearchCV(logit, est_dist_logit, n_iter)
grid_search_nb = RandomizedSearchCV(multi_nb, est_dist_nb, n_iter)
grid_search_sgd = RandomizedSearchCV(sgd, est_dist_sgd, n_iter)

#Get the estimators
clf1 = grid_search_rf.fit(X_train, y_train).best_estimator_
clf2 = grid_search_bst.fit(X_train, y_train).best_estimator_
clf3 = grid_search_logit.fit(X_train, y_train).best_estimator_
clf4 = grid_search_nb.fit(X_train, y_train).best_estimator_
clf5 = grid_search_sgd.fit(X_train, y_train).best_estimator_


estimators = [('rf', clf1), ('bst', clf2), ('bst', clf3), ('nb', clf4),
			  ('sgd', clf5)]



def main():
	print('We will begin to fit the models')
	print('Careful, this is going to take a while.' + 
	'Unless your computer is much more powerful than mine')
	
	#Get the estimators
	print('Tuning hyperparameters for the random forest model...')
	clf1 = grid_search_rf.fit(X_train, y_train).best_estimator_
	print('The results of the random forest model are:')
	print(confusion_matrix(y_test, clf1.predict(X_test)))
	print('Tuning hyperparameters for the gradient boosting model....')
	clf2 = grid_search_bst.fit(X_train, y_train).best_estimator_
	print('The results of the gradient boosting model are:')
	print(confusion_matrix(y_test, clf2.predict(X_test)))
	print('Tuning hyperparameters for logistic regression model....')
	clf3 = grid_search_logit.fit(X_train, y_train).best_estimator_
	print('The results of the logit model are:')
	print(confusion_matrix(y_test, clf3.predict(X_test)))
	print('Tuning hyperparameters for Naive Bayes model...')
	clf4 = grid_search_nb.fit(X_train, y_train).best_estimator_
	print('The results of the NB model are:')
	print(confusion_matrix(y_test, clf4.predict(X_test)))
	print('Tuning hyperparameters for Stochastic Gradient Descent (SGD) model....')
	clf5 = grid_search_sgd.fit(X_train, y_train).best_estimator_
	print('The results of the SGD model are:')
	print(confusion_matrix(y_test, clf5.predict(X_test)))
	
	print('Now we create an ensemble classifier from all five')
	
	
	#Create the ensemble classifier
	voter = VotingClassifier(estimators, voting = 'hard')
	voter.fit(X_train, y_train)
	
	#Despite the hours and hour of computation time, most of these models seem
	#to perform similarly and the confusion matrices indicate that they all have
	#a similar blindspot--a tendency to misclassify positive tweets as negative.
	#As such, the ensemble classifier really doesn't improve things as much as
	#might have liked. So it goes.

	#This is a bit of a 'throw everything at the wall and see what sticks'
	#kind of method, but (for me) this has really so far been about learning
	#how to deal with data of this size--keeping everything in a sparse matrix,
	#being careful about memory, etc... Like I mentioned above, the next step for me
	#will be actually digging into the data more
	print(voter.score(X_train, y_train))
	print(voter.score(X_test, y_test))
	
if __name__ == "__main__":
    main()