from nltk import word_tokenize

from NeuralNetworks import LanguageClassifier
import torch

model = LanguageClassifier(BATCH_SIZE=2048, SENTENCE_LENGTH=10, EMBEDDING_MODEL='./models/model_3_29.wordvectors')
model.load_state_dict(torch.load('./models/model_3_30.torch'))

# Prepare tensors to be fed into Neural Network with Batching
def prep_tensors(batch):
	# Create a list for label and tensors
	msg_tensor = []
	lbl_tensor = []
	
	
	
	# Convert the lists into tensors
	msg_tensor = torch.tensor(msg_tensor)
	lbl_tensor = torch.tensor(lbl_tensor)

	# Return the tensors
	return msg_tensor, lbl_tensor


if __name__ == "__main__":
	while True:
		# Gather input from the user
		user_sentence = input("Input a sample phrase: ")
	
		# Format the input into a set of tokens
		words = [token for token in word_tokenize(user_sentence.lower())]
		
		print('\t', words)
		
		# Turn the tokens into indices with the embedding model, normalize indices length
		indices = [model.embedding_model.wv.key_to_index[word] for word in words if word in model.embedding_model.wv.key_to_index]
		while len(indices) < 10:
			indices.append(0)
		if len(indices) > 10:
			indices = indices[:10]
		print('\t', indices)
		
		# Create a tensor from the indices to feed to the neural network
		tensor = torch.tensor([indices])
		print('\t', tensor)
		
		# Send these through the network
		output = model(tensor)
		
		print('\t', output)
	
	

