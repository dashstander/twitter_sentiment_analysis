#Twitter Sentiment Analysis Project

This is the (rough) beginning of my final project for Statistics 154: Machine Learning.

In the `_data` file you'll see the MaskedDataRaw.csv file that has approximately 1.5 million
tweets, their origin, and the "sentiment" associated with all but 50,000 of them. `0` represents
a negative sentiment, `1` represents a positive sentiment. `-1` indicates that a tweet is part
of the "test" set that will be used to judge the ultimate effectiveness of whatever model 
I build.

The full details of the project can be found at: 
			
			https://inclass.kaggle.com/c/tweetersentiment

In the _code folder there are currently two files: dealWithCommas.py and playingAround.py

These two files represent how I would generally go about exploring a dataset/problem and
what a more or less "first pass" at a solution would look like. What is not represented are the
different attempts at dimensionality reduction (PCA and even SVD were too memory intensive,
LASSO didn't select features better than Tfidf weighting, etc...) and model testing.

dealWithCommas.py is just the bare minimum of data-cleaning (in particular, the commas in the
tweets confused the csv format) so that I could load the data, and playingAround.py has me tuning
and testing various models, seeing how they work separately and in concert.

Not the most sophisticated analysis, but now I feel much more comfortable with the specifics
of the problem and am ready to dive into the data itself more.  