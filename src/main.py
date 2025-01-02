from embedding import load_embeddings

# Load the model and vocabulary
model, vocab = load_embeddings('../models/word2vec_embeddings.pt')

# Get embedding for a word
word = "example"
word_idx = vocab.get(word, vocab['<UNK>'])
embedding = model.get_embedding(word_idx)