#Formats tweets such that they can be read properly as a csv
import re


#Takes in data, an open file with 4 comma separated columns
#Makes sure that any further commas are safely quoted
#And the closes the file
def dwc(data):
	quote = re.compile('".*"')
	comma = re.compile(',')
	newLines = list()
	for line in data:
		#splits the line by commas
		fields = line.split(',', 3)
		#gets at what will be the last column
		tweet = fields[3]
		#takes off the new line stuff
		tweet = tweet.strip()
		
		#Is there a comma in the tweet?
		if tweet.count(',') != 0:
			#Are there quotes around it?
			isQuoted = quote.match(tweet)
			if isQuoted is None:
				#If there aren't, take out any quotes that are there
				#and put quotes around it
				tweet = tweet.replace('"', '')
				tweet = '"' + tweet + '"'
			tweet = tweet + "\r\n"
	
		else:
			#There aren't any commas, but there still might
			#be a quotation mark
			if tweet.count('"') != 0:
				tweet = tweet.replace('"', '')
			tweet = tweet + "\r\n"
			
		fields[3] = tweet
		newLines.append(",".join(fields))	
	return(newLines)
						
		 
	