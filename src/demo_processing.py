from dataset import CorneilleDataset
from utils.visualize import display_tokens

def main():
    # Create dataset instance
    dataset = CorneilleDataset('data/plays/melite.txt', sequence_length=5)
    
    # Display first 100 tokens
    print("\nProcessed Text Structure:")
    print("-" * 50)
    print(display_tokens(dataset.processed_text, max_tokens=100))
    
if __name__ == "__main__":
    main() 