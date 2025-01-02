import unittest
import torch
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import CorneilleDataset
from src.utils.visualize import display_tokens

class TestCorneilleDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a small test file and dataset instance."""
        # Create a temporary test file
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
        
        # Check act and scene marking
        self.assertIn('<ACT>', processed)
        self.assertIn('</ACT>', processed)
        self.assertIn('<SCENE>', processed)
        self.assertIn('</SCENE>', processed)
        
    def test_token_structure(self):
        """Test if tokens are properly structured and nested."""
        processed = self.dataset.processed_text[:100]
        formatted = display_tokens(processed)
        
        # Structure tests
        self.assertIn('<ACT>', processed)
        self.assertIn('</ACT>', processed)
        
        # Nesting tests
        self.assertTrue('<SCENE>' in formatted.split('<ACT>')[1])
        self.assertTrue('<CHAR>' in formatted.split('<SCENE>')[1])
        
    def test_sequence_creation(self):
        """Test if sequences are created correctly."""
        x, y = self.dataset[0]
        self.assertEqual(len(x), self.dataset.sequence_length)
        self.assertTrue(isinstance(y, int))
        
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