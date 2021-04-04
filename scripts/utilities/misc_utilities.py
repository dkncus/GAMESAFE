import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv
import numpy as np

"""
# Concatenate CSV's into one dataframe and split into train, test, and dev sets
scary = pd.read_csv('./messages_scary_pedo.csv')
lol = pd.read_csv('./messages_normal_lol.csv')
dota_sm = pd.read_csv('./messages_normal_dota_303k.csv')
dota_lg = pd.read_csv('./messages_normal_dota_huge.csv')
dota_lg_sample = dota_lg.sample(frac=0.0185, random_state = 200) # Take 400k messages from Dota Large Frame

frames = [scary, lol, dota_sm, dota_lg_sample]
df = pd.concat(frames)

print(len(df))

print('Sampling dataframes into train, dev, and test sets')
train, dev, test = np.split(df.sample(frac=1, random_state=42), 
                            [int(.98*len(df)), int(.99*len(df))]) 

print(len(train), type(train))
print(len(dev), type(dev))
print(len(test), type(test))

train.to_csv('./training/data_train.csv', index = False)
test.to_csv('./training/data_test.csv', index = False)
dev.to_csv('./training/data_dev.csv', index = False)
"""

"""
# Create edited window-sized messages for CSV's
WINDOW_SIZE = 10


window = pd.read_csv('./datasets/FORMATTEDdota2_chat_messages.csv')

messages = []

for i, row in window.iterrows():
	try:
		no_punct = ""
		for char in row['message']:
			if char not in punctuations:
				no_punct = no_punct + char
	
		# Tokenize the cleaned sentence
		words = word_tokenize(no_punct)
		
		# Add a row for every sliding window
		if len(words) >= WINDOW_SIZE:
			for ii in range(len(words) - WINDOW_SIZE):
				window = words[ii:ii + WINDOW_SIZE]
				string = ''
				for iii, str in enumerate(window):
					if iii < WINDOW_SIZE - 1:
						string += str + ' '
					else:
						string += str
					# print(i, row[2])
	
				messages.append({'Message': string, 'Type': 0})	
		elif len(words) >= 4:
			string = ''
			for iiii, str in enumerate(words):
				if iiii < WINDOW_SIZE - 1:
					string += str + ' '
				else:
					string += str
			
			messages.append({'Message': string, 'Type': 0})
			
		if i % 1000 == 0:
			print(i)
			
	except:
		print(i)

df_new = pd.DataFrame(messages, columns=['Message', 'Type'])
df_new.to_csv('./messages_normal_dota_window.csv', index = False)
"""
'''
View tensor data

			except Exception as e:
				print(data['message_words'])
				msg = msg_tensor.tolist()
				for obj in msg:
					x = dataset.embedding_model.wv.index2word[obj[0]]
					print(obj, x)
				print(lbl_tensor)
				print(e)
				exit(0)
'''

'''
LSTM Information
https://cnvrg.io/pytorch-lstm/?gclid=Cj0KCQjwjPaCBhDkARIsAISZN7S7uggC0XHu3gKn5jxi5YTgYF8Pu4JRf6yJS-nKXDdwGlfMjqCoTXoaAhUGEALw_wcB
'''

'''
# Concatenate 2 frames and save them as a CSV

df_1 = pd.read_csv('./messages_scary_with_window.csv')
df_2 = pd.read_csv('./messages_scary_window.csv')

frames = [df_1, df_2]

result = pd.concat(frames)

result.to_csv('./training/data_windowed.csv', index = False)
'''

"""
# Create a new CSV from the scary messages of messages 
stop_words = stopwords.words('english')  # Define list of stopwords

WINDOW_SIZE = 10

df = pd.read_csv('./longestfirst.csv', encoding='latin-1')
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

messages = []

# For each row in the CSV
for row in df.iterrows():
	# strip punctuation from sentence
	no_punct = ""
	for char in row[1]['Message']:
		if char not in punctuations:
			no_punct = no_punct + char
	
	# Tokenize the cleaned sentence
	words = word_tokenize(no_punct)
	
	# Add a row for every sliding window
	if len(words) >= WINDOW_SIZE:
		for i in range(len(words)- WINDOW_SIZE):
			window = words[i:i+WINDOW_SIZE]
			string = ''
			for i, str in enumerate(window):
				if i < WINDOW_SIZE - 1:
					string += str + ' '
				else: 
					string += str
			
			messages.append({'Message': string, 'Type': 1})	
			print(string)
	elif len(words) >= 4:
		string = ''
		for i, str in enumerate(words):
			if i < WINDOW_SIZE - 1:
				string += str + ' '
			else:
				string += str
		messages.append({'Message': string, 'Type': 1})	
		print(string)

df_new = pd.DataFrame(messages, columns=['Message', 'Type'])
df_new.to_csv('./messages_scary_window.csv', index = False)
print(len(messages))
"""

'''
# Get the dataframe sorted by longest message

df = pd.read_csv('./datasets/FORMATTEDLOLChatLog.csv', encoding='latin-1')
x = df.reindex(df['message'].str.len().sort_values().index)
x = x.reindex(index = x.index[::-1])
x.to_csv('./longestfirst_LoL.csv', index = False)
'''

'''
for splitting datasets amongst various .csv files

import pandas as pd

df = pd.read_csv('./training/data_full_clean.csv')

train = df.sample(frac=0.98,random_state = 200) #random state is a seed value
devtest = df.drop(train.index)
test = devtest.sample(frac=0.5, random_state=200)
dev = devtest.drop(test.index)

train.to_csv('./training/data_train.csv', index = False)
test.to_csv('./training/data_test.csv', index = False)
dev.to_csv('./training/data_dev.csv', index = False)
'''
