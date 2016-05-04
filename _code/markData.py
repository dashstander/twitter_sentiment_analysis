import pandas as pd
import numpy as np
import sklearn as skl
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
import nltk
import xgboost as xgb
from scipy import sparse, io

sentiment_df = pd.read_csv("_data/MaskedDataUsable.csv", quotechar = '"',
						   skipinitialspace = True)
						   
	
tweets = sentiment_df.SentimentText
sentiment = sentiment_df.Sentiment
test_data = (sentiment == -1)
tweets = tweets[-test_data]
sentiment = sentiment[-test_data]

#RegEx Strings
happy_emoji = r'[:=;X][\-oO]?[\)D\]pPO\/\*]'
sad_emoji = r'[:=;X][\-oO][\(|\[]'
exclamation = r"!"
question = r"\?"
mention = r"@[a-z-A-Z0-9_]+"
URL =  r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
hashtags = "#"
heart = '&lt;3'
quote_ascii = "&quot"
quote_basic = "'"



def tagNots(tweet):
	negate_pattern = re.compile(r'not ?|isnt ?|wont ?|dont ?|cant ?|shant ?|mustnt ?|couldnt ?|wouldnt ?|shouldnt ?')
	end_phrase_pattern = re.compile(r'\.|,|;')
	negate_match = re.search(negate_pattern, tweet)
	end_phrase_match = re.finditer(end_phrase_pattern, tweet)
	space = ' '	
	if (negate_match is None):
		return(tweet)
		
	else:
		end = None
		for w in end_phrase_match:
			if w.start(0) > negate_match.start(0):
				end = w.start(0)
				break;
		
		if (end is None):
			pos = negate_match.end(0)
			begin_tweet = tweet[:pos]
			end_tweet = ''
			phrase =  tweet[pos:]
		else:
			begin = negate_match.start(0)
			begin_tweet = tweet[:begin]
			end_tweet = tweet[end:]
			phrase = tweet[begin:end]
		tokenized_phrase = nltk.word_tokenize(phrase)
		i = 0
		for w in tokenized_phrase:
			if (i != 0):
				tokenized_phrase[i] = 'not' + w
			i += 1
			
		tokenized_phrase.insert(0, begin_tweet)
		tokenized_phrase.append(end_tweet)
		return(space.join(tokenized_phrase))
	


def textProcessing(tweet):
	'Takes a tweet and processes it '
	
	#RegEx Strings
	happy_emoji = r"[:=;X][\-oO]?[\)D\]pPO\\/\*]"
	sad_emoji = r"[:=;X][\-oO]?[\(\|\[]"
	exclamation = r"!"
	question = r"\?"
	mention = r"@[a-z-A-Z0-9_]+"
	URL =  r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
	hashtags = "#"
	heart = '&lt;3'
	quote_ascii = "&quot"
	quote_basic = "'"
	amp = '&amp;'
	ellipsis = '\.{3,}'

		 
	patterns = [URL, happy_emoji, sad_emoji, exclamation, question, mention,
				hashtags, heart, quote_ascii, quote_basic, amp, ellipsis]
	replacements = [" URL ", 'HAPPYEMOJI ', 'SADEMOJI ', " EXCLAMATION ", " QUESTION ",
					'MENTION ', "",' HEART ', "", "", "and", "ELLIPSIS"]
	
	
	for i in range(0,(len(patterns)-1)):
		tweet = re.sub(patterns[i], replacements[i], tweet)
		
	tweet = tagNots(tweet)	
	
	return tweet;
	
tfidf = TfidfVectorizer(strip_accents = "ascii", 
                        ngram_range = (1,2),
                        max_features = 10000, 
                        stop_words = 'english', 
                        max_df = .95)



newTweets = tweets.apply(textProcessing)
tweet_dtm = tfidf.fit_transform(newTweets)

X_train, X_test, y_train, y_test = train_test_split(tweet_dtm, sentiment, test_size = .333)

params = {'objective':'binary:logistic', 'eta':.8, 'gamma':5, 'max_depth':20, 'verbose':3, 
			'eval_metric':['auc', 'error']}
num_rounds = 250

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)

watchlist = [(dtrain, 'train'), (dtest, 'eval')]


def main():
	print(dict(zip(tfidf.get_feature_names(), tfidf.idf_)))
	dtrain.save_binary('_data/XGB_Train_DTM')
	dtest.save_binary('_data/XGB_Test_DTM')

if __name__ == "__main__":
	main()
	
	

