from torch.utils.data import Dataset
import torch
import re

class CorneilleDataset(Dataset):
    def __init__(self, text_path: str, sequence_length: int, vocab=None, transform=None):
        """
        Args:
            text_path (str): Path to the play file
            sequence_length (int): Length of sequences to generate
            vocab (dict, optional): Vocabulary mapping
            transform (callable, optional): Optional transform to be applied
        """
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Define regex patterns for text processing
        self.CHAR_PATTERN = r'^([A-Z]+)\.'
        self.INDENT_PATTERN = r'^\s+'
        
        # Load and preprocess text
        with open(text_path, 'r', encoding='utf-8') as f:
            self.raw_text = f.read()
            
        # Initialize vocabulary if not provided
        self.vocab = vocab if vocab is not None else self._build_vocab()
        
        # Process text with structural tokens
        self.processed_text = self._process_text(self.raw_text)
        
        # Create sequences
        self.sequences = self._create_sequences()
        
    def _build_vocab(self):
        """Build vocabulary from text, including special tokens."""
        special_tokens = {
            '<PAD>': 0,      # For padding sequences
            '<UNK>': 1,      # For unknown words
            '<LINE>': 2,     # Opening line tag
            '</LINE>': 3,    # Closing line tag
            '<ACT>': 4,      # Opening act tag
            '</ACT>': 5,     # Closing act tag
            '<SCENE>': 6,    # Opening scene tag
            '</SCENE>': 7,   # Closing scene tag
            '<CHAR>': 8,     # Opening character speech tag
            '</CHAR>': 9,    # Closing character speech tag
        }
        
        # Create word frequency counter
        word_freq = {}
        for word in self.raw_text.split():
            word = word.lower()  # Convert to lowercase
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Create vocabulary with words appearing more than min_freq times
        vocab = special_tokens.copy()
        idx = len(special_tokens)
        min_freq = 2  # Minimum frequency threshold
        
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
            if freq >= min_freq:
                vocab[word] = idx
                idx += 1
                
        self.idx_to_word = {idx: word for word, idx in vocab.items()}
        return vocab
    
    def _process_text(self, raw_text: str) -> list:
        """Process raw text into structured format with XML-like tags.
        
        Args:
            raw_text (str): Raw input text
            
        Returns:
            list: List of tokens including structural markers
        """
        lines = raw_text.split('\n')
        processed_tokens = []
        current_speaker = None
        in_speech = False
        
        for line in lines:
            if not line.strip():  # Skip empty lines
                continue
                
            # Check for character name
            char_match = re.match(self.CHAR_PATTERN, line.strip())
            if char_match:
                if in_speech:
                    processed_tokens.append('</CHAR>')
                current_speaker = char_match.group(1)
                processed_tokens.extend(['<CHAR>', current_speaker])
                in_speech = True
                continue
                
            # Process regular line
            cleaned_line = line.strip()
            if cleaned_line:
                processed_tokens.append('<LINE>')
                processed_tokens.extend(cleaned_line.split())
                processed_tokens.append('</LINE>')
        
        if in_speech:
            processed_tokens.append('</CHAR>')
            
        return processed_tokens
    
    def _create_sequences(self) -> list:
        """Create sequences for training.
        
        Returns:
            list: List of sequences, each being a list of token indices
        """
        sequences = []
        tokens = self.processed_text
        
        # Convert tokens to indices
        token_indices = [self.vocab.get(token.lower(), self.vocab['<UNK>']) 
                        for token in tokens]
        
        # Create sequences of length sequence_length + 1 
        # (+1 for the target word)
        for i in range(len(token_indices) - self.sequence_length):
            sequence = token_indices[i:i + self.sequence_length + 1]
            sequences.append(sequence)
            
        return sequences
    
    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Return a single training example.
        
        Args:
            idx (int): Index of the sequence
            
        Returns:
            tuple: (input_sequence, target_word)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sequence = self.sequences[idx]
        
        # Split into input and target
        x = sequence[:-1]  # all but last token
        y = sequence[-1]   # last token is target
        
        if self.transform:
            x = self.transform(x)
            
        return x, y 