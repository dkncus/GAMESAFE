# NLTK Natural Language Processing Models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import gensim
from gensim.models import Word2Vec
import csv

class NLPModelGenerator():
	# Initialization Method
	def __init__(self, train=False, load=True):
		# Train the model
		if train:
			print('Loading Messages...')
			messages = self.load_csv_to_list('./datasets/data_full_clean.csv')
			print('Creating embedding model...')
			self.model = self.create_model(messages)
			self.model = self.pickle_model(self.model, './models/model_3_29.wordvectors')
			print('Model compiled and saved.')
		# Load a preexisting model
		if load:
			print('Loading preexisting model...')
			self.model = gensim.models.KeyedVectors.load('./models/model.wordvectors')
			print('Model loaded.')
		
	# Create a list of messages from a CSV file.
	def load_csv_to_list(self, csv_location):
		print('Reading File', csv_location)
	
		# List of messages to return
		messages = []
	
		# Load dataset
		with open(csv_location, newline='', encoding='utf-8', errors='ignore') as sample:
	
			# Create a CSV reader object delimited at commas
			reader = csv.reader(sample, delimiter=',')
			
			for i, row in enumerate(reader):
				# print(i, row[2])
				messages.append(row[0])  # Read into message list
	
		return messages
	
	# Create and return a trained model.
	def create_model(self, messages):
		# Create data object (Lists of words)
		data = []
	
		# For each message
		num_messages = len(messages)
		for i, message in enumerate(messages):
			# Tokenize the sentence into a list of words
			temp = [token for token in word_tokenize(message)]
	
			# Append a list of words as a message to the dataset
			data.append(temp)
			
			if i % 10000 == 0:
				# Diagnostic Information
				print(f'[{i}/{num_messages}] Messages Loaded')
	
		# Create the word embedding model with Word2Vec algorithm
		print("Generating word embedding model...")
		model = gensim.models.Word2Vec(data, min_count=20, vector_size=16, window=10, sg=1, workers=3)
		print("Embedding model created.")
	
		return model
	
	# Save the model to a file
	def pickle_model(self, model, filename):
		model.save(fname_or_handle=filename)
		return model

if __name__ == '__main__':
	# List of datasets
	nlp = NLPModelGenerator(train=True, load=False)
	
	# For each of these words
	words = ['lol','talk', 'game']

	# Print diagnostic data about words similar to this word
	for word in words:
		print('Analysis for', word)
		print(nlp.model.wv.index_to_key[word])