import unittest
import torch
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import CorneilleDataset
from src.dataloader import ContextualDataLoader

class TestContextualDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a small test file and dataset instance."""
        cls.test_text = """
ACTE I

SCÈNE I

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
        
    def test_dataloader_creation(self):
        """Test if dataloader can be created with basic parameters."""
        dataloader = ContextualDataLoader(
            self.dataset,
            batch_size=2,
            sequence_length=5
        )
        self.assertIsNotNone(dataloader)
        
    def test_batch_generation(self):
        """Test if dataloader produces valid batches."""
        dataloader = ContextualDataLoader(
            self.dataset,
            batch_size=2,
            sequence_length=5
        )
        
        # Get first batch
        for batch in dataloader:
            x, y = batch
            # Check shapes
            self.assertEqual(x.dim(), 2)  # [batch_size, sequence_length]
            self.assertEqual(y.dim(), 1)  # [batch_size]
            self.assertEqual(x.size(1), 5)  # sequence_length
            break
            
    def test_line_integrity(self):
        """Test if sequences maintain line integrity."""
        dataloader = ContextualDataLoader(
            self.dataset,
            batch_size=1,
            sequence_length=10
        )
        
        for batch in dataloader:
            x, y = batch
            # Convert indices back to tokens
            tokens = [self.dataset.idx_to_word[idx.item()] for idx in x[0]]
            
            # Check if any line is split
            line_starts = [i for i, t in enumerate(tokens) if t == '<LINE>']
            line_ends = [i for i, t in enumerate(tokens) if t == '</LINE>']
            
            # Each start should have a corresponding end
            self.assertEqual(len(line_starts), len(line_ends))
            for start, end in zip(line_starts, line_ends):
                self.assertTrue(start < end)
            break
            
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