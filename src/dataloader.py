from torch.utils.data import DataLoader
import torch
from typing import List, Tuple

class ContextualDataLoader(DataLoader):
    def __init__(self, 
                 dataset,
                 batch_size: int,
                 sequence_length: int,
                 shuffle: bool = True):
        """
        Custom DataLoader that creates sequences respecting text structure.
        
        Args:
            dataset: CorneilleDataset instance
            batch_size (int): Number of sequences per batch
            sequence_length (int): Length of sequences to generate
            shuffle (bool): Whether to shuffle the sequences
        """
        self.sequence_length = sequence_length
        self.vocab = dataset.vocab
        self.idx_to_word = dataset.idx_to_word
        
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )
        
    def _collate_fn(self, batch: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sequences from batch of token indices.
        
        Args:
            batch: List of token indices from the dataset
            
        Returns:
            tuple: (input_sequences, target_sequences)
        """
        all_sequences = []
        
        # Process each chunk of tokens
        for tokens in batch:
            # Create sequences of length sequence_length + 1
            for i in range(len(tokens) - self.sequence_length):
                sequence = tokens[i:i + self.sequence_length + 1]
                
                # Only add sequence if it doesn't split a line
                tokens_str = [self.idx_to_word[idx] for idx in sequence]
                line_starts = [i for i, t in enumerate(tokens_str) if t == '<LINE>']
                line_ends = [i for i, t in enumerate(tokens_str) if t == '</LINE>']
                
                if len(line_starts) == len(line_ends):
                    all_sequences.append(sequence)
                
                if len(all_sequences) >= self.batch_size:
                    break
            
            if len(all_sequences) >= self.batch_size:
                break
        
        if all_sequences:
            # Convert to tensors and ensure proper shape
            sequences = torch.tensor(all_sequences)  # [batch_size, seq_len + 1]
            x = sequences[:, :-1]  # [batch_size, seq_len]
            y = sequences[:, -1]   # [batch_size]
            
            # Ensure we have the right batch size
            if x.size(0) < self.batch_size:
                pad_size = self.batch_size - x.size(0)
                x = torch.cat([x, torch.zeros(pad_size, self.sequence_length).long()], dim=0)
                y = torch.cat([y, torch.zeros(pad_size).long()], dim=0)
            
            return x[:self.batch_size], y[:self.batch_size]
        
        # Return properly shaped empty tensors if no sequences
        return (torch.zeros(self.batch_size, self.sequence_length).long(),
                torch.zeros(self.batch_size).long()) 