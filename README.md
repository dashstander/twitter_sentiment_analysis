#Twitter Sentiment Analysis Project

This is my final project for Statistics 154: Machine Learning.

In the `_data` file you'll see the MaskedDataRaw.csv file that would have approximately 1.5 million
tweets, their origin, and the "sentiment" associated with all but 50,000 of them. However, that is too much for github to handle. They can be found, along with the full details of the project, at: 
			
			https://inclass.kaggle.com/c/tweetersentiment

In the _code folder there are a number of files:

- dealWithCommas.py transforms the raw data such that it cand be loaded as a pandas.DataFrame into python
- markData.py performs the feature engineering creates the Document-Term-Matrix
- playingAround.py is me exploring, and trying out many different models
- boostCV.py is where I wrote a wrapper to perform cross-validation with xgboost
- hierarchicalModel.py is my attempt to make a hierarchical model using both xgboost and logistic regression


