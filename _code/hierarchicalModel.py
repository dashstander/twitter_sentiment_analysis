import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from scipy import sparse, io

tweet_dtm = io.mmread('_data/sparse_dtm.mtx')
sentiment_df = pd.read_csv("_data/MaskedDataUsable.csv", quotechar = '"',
						   skipinitialspace = True)
						   
sentiment = sentiment_df.Sentiment
test_data = sentiment == -1
sentiment = sentiment[-test_data]




X_train, X_test, y_train, y_test = train_test_split(tweet_dtm, sentiment, test_size = .333)

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)




params = {'objective':'binary:logistic', 'eta':.65, 'gamma':5, 'max_depth':20, 'verbose':3, 
			'eval_metric':['auc', 'error']}
num_rounds = 200
watchlist = [(dtrain, 'train'), (dtest, 'eval')]

bst = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds = 3)


logit_pos = lm.LogisticRegression(C= .6, solver = 'sag', class_weight = 'balanced',
								verbose = 3, warm_start = True)
								
logit_neg = lm.LogisticRegression(C= .6, solver = 'sag', class_weight = 'balanced',
								verbose = 3, warm_start = True)
#logit_pos.fit(X_train, y_train)
#logit_neg.fit(X_train, y_train)

bst_train_preds = bst.predict(dtrain) > .5
bst_test_preds = bst.predict(dtest) > .5




train_pos = np.where(bst_train_preds)[0]
train_neg = np.where(bst_train_preds == 0)[0]

test_pos =  np.where(bst_test_preds)[0]
test_neg =  np.where(bst_test_preds == 0)[0]

X_train_pos = X_train[train_pos, :]
y_train_pos = y_train.as_matrix()[train_pos]

X_train_neg = X_train[train_neg, :]
y_train_neg = y_train.as_matrix()[train_neg]

X_test_pos = X_test[test_pos, :]
y_test_pos = y_test.as_matrix()[test_pos]

X_test_neg = X_test[test_neg, :]
y_test_neg = y_test.as_matrix()[test_neg]


logit_pos.fit(X_train_pos, y_train_pos)
logit_neg.fit(X_train_neg, y_train_neg)

















