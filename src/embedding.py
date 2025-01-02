"""
Word2Vec Embedding Module

This module implements a Word2Vec model using PyTorch for creating word embeddings
from text data. It includes functionality for:
- Text preprocessing
- Vocabulary building
- Skip-gram model training
- Embedding visualization

The model can be saved and loaded for use in broader NLP pipelines.

Typical usage:
    # Training
    python embedding.py

    # Using in another project
    from embedding import Word2Vec, load_embeddings
    model, vocab = load_embeddings('path/to/word2vec_embeddings.pt')

Author: [Your Name]
Date: [Current Date]
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import os
import plotly.graph_objects as go

class Word2Vec(nn.Module):
    """
    Skip-gram Word2Vec model implementation.
    
    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of the word embeddings
    
    Attributes:
        embeddings (nn.Embedding): Embedding layer
        linear (nn.Linear): Linear layer for context prediction
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        embeds = self.embeddings(inputs)
        output = self.linear(embeds)
        return output

    def get_embedding(self, word_idx: int) -> np.ndarray:
        """Get the embedding vector for a word index."""
        return self.embeddings.weight[word_idx].detach().numpy()

class SkipGramDataset(Dataset):
    """Dataset for skip-gram model training."""
    def __init__(self, pairs: List[Tuple[int, int]]):
        self.pairs = pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.pairs[idx][0]), torch.tensor(self.pairs[idx][1])

def preprocess_text(text: str) -> List[str]:
    """
    Clean and tokenize text into words, preserving Unicode characters.
    
    Args:
        text (str): Input text to preprocess
    
    Returns:
        List[str]: List of preprocessed tokens
    """
    # Convert to lowercase
    text = text.lower()
    # Split into words, preserving Unicode characters
    words = re.findall(r'\b[\w\u00C0-\u017F]+\b', text, re.UNICODE)
    return words

def build_vocabulary(words: List[str], min_count: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create word-to-index and index-to-word mappings.
    
    Args:
        words (List[str]): List of words to build vocabulary from
        min_count (int): Minimum frequency for a word to be included
    
    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: Word-to-index and index-to-word mappings
    """
    word_counts = Counter(words)
    vocabulary = {word: idx + 1 for idx, (word, count) 
                 in enumerate(word_counts.items()) 
                 if count >= min_count}
    vocabulary['<UNK>'] = 0  # Unknown token
    
    idx_to_word = {idx: word for word, idx in vocabulary.items()}
    return vocabulary, idx_to_word

def create_skip_gram_pairs(words: List[str], 
                         vocabulary: Dict[str, int],
                         window_size: int = 5) -> List[Tuple[int, int]]:
    """Generate training pairs for skip-gram model."""
    pairs = []
    for i, word in enumerate(words):
        word_idx = vocabulary.get(word, vocabulary['<UNK>'])
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        
        for j in range(start, end):
            if i != j:
                context_word = words[j]
                context_idx = vocabulary.get(context_word, vocabulary['<UNK>'])
                pairs.append((word_idx, context_idx))
    
    return pairs

def load_embeddings(model_path: str) -> Tuple[Word2Vec, Dict[str, int]]:
    """
    Load a trained Word2Vec model and its vocabulary.
    
    Args:
        model_path (str): Path to the saved model checkpoint
    
    Returns:
        Tuple[Word2Vec, Dict[str, int]]: Loaded model and vocabulary
    """
    checkpoint = torch.load(model_path)
    model = Word2Vec(len(checkpoint['vocabulary']), checkpoint['embedding_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['vocabulary']

def visualize_embeddings(model_path: str, 
                        dim1: int = 2, 
                        dim2: int = 3, 
                        top_n: int = None, 
                        save_plot: bool = False) -> None:
    """
    Visualize word embeddings from a trained Word2Vec model using Plotly.
    
    Args:
        model_path: Path to the saved model checkpoint
        dim1: First dimension to plot (0-based index)
        dim2: Second dimension to plot (0-based index)
        top_n: If set, only visualize this many of the most frequent words
        save_plot: If True, saves the plot as HTML instead of displaying
    """
    import plotly.graph_objects as go
    
    # Load the trained model
    checkpoint = torch.load(model_path)
    model = Word2Vec(len(checkpoint['vocabulary']), checkpoint['embedding_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get embeddings
    embeddings = model.embeddings.weight.detach().numpy()
    vocab = checkpoint['vocabulary']
    
    # Validate dimensions
    if max(dim1, dim2) >= embeddings.shape[1]:
        raise ValueError(f"Requested dimensions {dim1} and {dim2} must be less than embedding dimension {embeddings.shape[1]}")
    
    # Filter for top N most frequent words if specified
    if top_n is not None:
        # Sort vocabulary by index (lower index = more frequent)
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        # Take only top N items (excluding UNK token)
        vocab_items = [item for item in sorted_vocab if item[0] != '<UNK>'][:top_n]
    else:
        vocab_items = vocab.items()
    
    # Prepare data for plotting
    x_coords = []
    y_coords = []
    text_labels = []
    
    for word, idx in vocab_items:
        x_coords.append(embeddings[idx, dim1])
        y_coords.append(embeddings[idx, dim2])
        text_labels.append(word)
    
    # Create the plot
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers+text',
        text=text_labels,
        textposition="top center",
        marker=dict(size=8)
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Word Embeddings Visualization (Dimensions {dim1+1} and {dim2+1})<br>{"Top "+str(top_n)+" words" if top_n else "All words"}',
        xaxis_title=f"Dimension {dim1+1}",
        yaxis_title=f"Dimension {dim2+1}",
        width=800,
        height=800,
        showlegend=False
    )
    
    if save_plot:
        fig.write_html(f"embeddings_visualization_dim{dim1+1}_{dim2+1}.html")
    else:
        fig.show()

def main():
    """Train the Word2Vec model on the input text."""
    # Load the text file
    try:
        with open('data/plays/melite.txt', 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("Error: Could not find the input text file.")
        return
    except UnicodeDecodeError:
        print("Error: File encoding issues. Make sure the file is UTF-8 encoded.")
        return

    # Process the loaded content
    words = preprocess_text(content)
    vocabulary, idx_to_word = build_vocabulary(words)
    vocab_size = len(vocabulary)
    
    # Create training pairs
    pairs = create_skip_gram_pairs(words, vocabulary)
    
    # Create dataset and dataloader
    dataset = SkipGramDataset(pairs)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Initialize model parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_dim = 100
    
    # Try to load existing model
    start_epoch = 0
    model_path = 'word2vec_embeddings.pt'
    if os.path.exists(model_path):
        print("Loading existing model...")
        checkpoint = torch.load(model_path)
        model = Word2Vec(vocab_size, checkpoint['embedding_dim']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        embedding_dim = checkpoint['embedding_dim']
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Creating new model...")
        model = Word2Vec(vocab_size, embedding_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    num_epochs = 20
    for epoch in range(start_epoch, start_epoch + num_epochs):
        total_loss = 0
        for batch_idx, (input_words, target_words) in enumerate(dataloader):
            input_words = input_words.to(device)
            target_words = target_words.to(device)
            
            # Forward pass
            outputs = model(input_words)
            loss = criterion(outputs, target_words)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss/len(dataloader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
        
        # Save checkpoint after each epoch with current epoch number
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocabulary': vocabulary,
            'idx_to_word': idx_to_word,
            'embedding_dim': embedding_dim,
            'epoch': epoch + 1  # Save the current epoch number
        }, model_path)
    
    # Comment out or remove the visualization call
    # visualize_embeddings('word2vec_embeddings.pt', top_n=100)

if __name__ == "__main__":
    main()

