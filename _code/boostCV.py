import xgboost as xgb
import pandas as pd
import numpy as np
import scipy as sp
from scipy import sparse, io
from sklearn.cross_validation import train_test_split

def xgbCV(dmatrix,  nfolds, eta_list, gamma_list, num_rounds = 500):
	
	params = {'eta':'', 'gamma':'', 'objective':'binary:logistic', 'verbose':3,
				'max_depth':20, 'subsample':.75, 'colsample_bytree':.75}
	
	vals = {'eta':[], 'gamma':[], 'num_iter':[], 'mean_cv_error':[], 'std_cv_error':[]}
	
	
	for e in eta_list:
		for g in gamma_list:
			params['eta'] = e
			params['gamma'] = g
			
			vals['eta'].append(e)
			vals['gamma'].append(g)
			
			print('Training the booster with a learning rate of', e, "and gamma of ", g)
			bst = xgb.cv(params, dmatrix, num_rounds, nfolds, early_stopping_rounds = 2)
			print('Stopped after', len(bst.index), "rounds.")
			
			best_iter = bst.nsmallest(1, 'test-error-mean')
			vals['num_iter'].append(best_iter.index[0])
			vals['mean_cv_error'].append(best_iter['test-error-mean'])
			vals['std_cv_error'].append(best_iter['test-error-std'])
			
	cv_df = pd.DataFrame.from_dict(vals)
	
	return(cv_df)
	
twitter_df = pd.read_csv("_data/MaskedDataUsable.csv",
                         quotechar = '"', skipinitialspace = True)                         
sentiment = twitter_df.Sentiment
test_data = (sentiment == -1)
sentiment = sentiment[-test_data]


dtrain = xgb.DMatrix('_data/XGB_Train_DTM')
dtest = xgb.DMatrix('_data/XGB_Test_DTM')


eta_list = np.linspace(.1, .9, 25)
gamma_list = [4, 5, 6, 7]
nfolds = 3

cv_df = xgbCV(dtrain, nfolds, eta_list, gamma_list)

def main():
	print(cv_df)
			

if __name__ == '__main__':
	main()
	
	
