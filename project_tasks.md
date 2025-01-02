# Project Tasks Tracker

## 🎯 Embedding & Vocabulary Tasks
1. Add special tokens for text structure:
   - [ ] `<ACT>` - Mark beginning of acts
   - [ ] `<SCENE>` - Mark beginning of scenes
   - [ ] `<CHAR>` - Mark character names/dialogue attribution
   - [ ] `<STAGE>` - Mark stage directions
   - [ ] `<PAD>` - For padding sequences
   - [ ] `<BOS>` - Beginning of sequence
   - [ ] `<EOS>` - End of sequence
   - [x] `<UNK>` - Unknown tokens (already implemented)

2. Modify embedding handling:
   - [ ] Update vocabulary building to include special tokens
   - [ ] Ensure special tokens are not used in context windows
   - [ ] Retrain embeddings with new tokens

## 📝 Text Processing Tasks
1. Parse play structure:
   - [ ] Identify and mark act boundaries
   - [ ] Identify and mark scene boundaries
   - [ ] Extract character names and dialogue
   - [ ] Identify stage directions

2. Create data preprocessing pipeline:
   - [ ] Clean and normalize text
   - [ ] Handle French-specific characters
   - [ ] Create sequence windows for training
   - [ ] Implement padding strategy

## 🔄 Dataset & DataLoader Tasks
1. Create custom Dataset class:
   - [ ] Implement sequence generation
   - [ ] Handle special tokens properly
   - [ ] Implement proper text windowing

2. Create DataLoader:
   - [ ] Define batch creation logic
   - [ ] Implement padding for batches
   - [ ] Add data augmentation if needed

## 🤖 Model Tasks
1. Design next word prediction architectures (progressive complexity):
   - [ ] Basic RNN implementation:
     - [ ] Single layer RNN
     - [ ] Define input/output dimensions
     - [ ] Add dropout
   - [ ] Multi-layer RNN variants:
     - [ ] Implement deeper architecture
     - [ ] Add residual connections
     - [ ] Compare performance
   - [ ] GRU/LSTM implementations:
     - [ ] Convert RNN to GRU
     - [ ] Convert RNN to LSTM
     - [ ] Compare gates' impact
   - [ ] Transformer implementation:
     - [ ] Basic transformer block
     - [ ] Multi-head attention
     - [ ] Position encoding

2. Architecture comparison framework:
   - [ ] Define comparison metrics
   - [ ] Create standardized evaluation pipeline
   - [ ] Implement logging for model comparisons
   - [ ] Create visualization tools for comparing results

3. Training setup:
   - [ ] Define loss function
   - [ ] Choose optimizer
   - [ ] Set up training loop
   - [ ] Add validation step
   - [ ] Implement early stopping

## 📊 Evaluation Tasks
1. Define metrics:
   - [ ] Perplexity
   - [ ] Accuracy
   - [ ] Custom metrics for theater-specific evaluation

2. Create evaluation pipeline:
   - [ ] Split data into train/val/test
   - [ ] Create evaluation scripts
   - [ ] Set up logging

## 🧪 Testing Tasks
1. Create unit tests:
   - [ ] Test data loading
   - [ ] Test preprocessing
   - [ ] Test model components

## 📚 Documentation Tasks
1. Create documentation:
   - [ ] Add docstrings
   - [ ] Create README
   - [ ] Document model architecture
   - [ ] Add usage examples 