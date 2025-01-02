import unittest
import torch
from src.dataset import CorneilleDataset

class TestCorneilleDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a small test file and dataset instance."""
        # Create a temporary test file
        cls.test_text = """
TIRCIS.

Ne dissimulons point: tu règles mieux ta flamme,
Et tu n'es pas si fou que d'en faire ta femme.

ÉRASTE.

Quoi! tu sembles douter de mes intentions?
"""
        with open('tests/test_play.txt', 'w', encoding='utf-8') as f:
            f.write(cls.test_text)
            
        # Create dataset instance
        cls.dataset = CorneilleDataset('tests/test_play.txt', sequence_length=5)
        
    def test_vocabulary_creation(self):
        """Test if vocabulary contains all special tokens."""
        expected_tokens = {
            '<PAD>', '<UNK>', '<LINE>', '</LINE>',
            '<ACT>', '</ACT>', '<SCENE>', '</SCENE>',
            '<CHAR>', '</CHAR>'
        }
        vocab_tokens = set(self.dataset.vocab.keys())
        for token in expected_tokens:
            self.assertIn(token, vocab_tokens)
            
    def test_text_processing(self):
        """Test if text is properly processed with tokens."""
        processed = self.dataset.processed_text
        
        # Check character marking
        self.assertIn('<CHAR>', processed)
        self.assertIn('TIRCIS', processed)
        self.assertIn('</CHAR>', processed)
        
        # Check line marking
        self.assertIn('<LINE>', processed)
        self.assertIn('</LINE>', processed)
        
    def test_sequence_creation(self):
        """Test if sequences are created correctly."""
        # Check sequence length
        x, y = self.dataset[0]
        self.assertEqual(len(x), self.dataset.sequence_length)
        self.assertTrue(isinstance(y, int))
        
    def test_dataset_length(self):
        """Test if dataset length is correct."""
        expected_length = len(self.dataset.sequences)
        self.assertEqual(len(self.dataset), expected_length)
        
    def test_getitem(self):
        """Test if __getitem__ returns correct format."""
        x, y = self.dataset[0]
        self.assertTrue(isinstance(x, list))
        self.assertTrue(isinstance(y, int))
        self.assertEqual(len(x), self.dataset.sequence_length)
        
    def test_unknown_words(self):
        """Test handling of unknown words."""
        # Create dataset with minimal vocab
        small_vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<LINE>': 2,
            '</LINE>': 3,
        }
        dataset = CorneilleDataset('tests/test_play.txt', 
                                 sequence_length=5, 
                                 vocab=small_vocab)
        
        # Check if unknown words are mapped to UNK token
        x, y = dataset[0]
        self.assertIn(1, x)  # 1 is the index for <UNK>
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        import os
        try:
            os.remove('tests/test_play.txt')
        except:
            pass

if __name__ == '__main__':
    unittest.main() 