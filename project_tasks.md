# Project Tasks Tracker

## 🎯 Embedding & Vocabulary Tasks
1. Add special tokens for text structure:
   - [x] `<LINE>`, `</LINE>` - Mark beginning/end of lines
   - [x] `<ACT>`, `</ACT>` - Define and implement tokens
   - [x] `<SCENE>`, `</SCENE>` - Define and implement tokens
   - [x] `<CHAR>`, `</CHAR>` - Mark character speeches
   - [x] `<PAD>` - For padding sequences (not yet used)
   - [x] `<UNK>` - Unknown tokens

2. Modify embedding handling:
   - [x] Update vocabulary building to include special tokens
   - [ ] Ensure special tokens are not used in context windows
   - [x] Implement 100-dimension embeddings

## 📝 Text Processing Tasks
1. Parse play structure:
   - [x] Extract character names and dialogue
   - [x] Identify and mark act boundaries
     - [x] Detect "ACTE" markers in text
     - [x] Insert `<ACT>`, `</ACT>` tokens
     - [x] Validate act structure
   - [x] Identify and mark scene boundaries
     - [x] Detect "SCÈNE" markers in text
     - [x] Insert `<SCENE>`, `</SCENE>` tokens
     - [x] Validate scene structure
   - [ ] Identify stage directions
     - [ ] Detect parenthetical text
     - [ ] Handle italicized directions

2. Create data preprocessing pipeline:
   - [x] Clean and normalize text
   - [ ] Handle French-specific characters
     - [ ] Proper handling of accents (é, è, ê, etc.)
     - [ ] Handle special punctuation (œ, «, », etc.)
   - [-] Create sequence windows for training
     - [x] Basic sequence creation
     - [ ] Respect line boundaries
     - [ ] Preserve rhyming context (4-line minimum)
   - [ ] Implement padding strategy
     - [ ] Add padding for batch consistency
     - [ ] Handle variable length sequences

## 🔄 Dataset & DataLoader Tasks
1. Create custom Dataset class:
   - [-] Implement sequence generation
     - [x] Basic sequence splitting
     - [ ] Context-aware windowing
     - [ ] Rhyme pattern preservation
   - [-] Handle special tokens properly
     - [x] Insert line and character tokens
     - [ ] Insert act and scene tokens
     - [ ] Validate token nesting
   - [ ] Implement proper text windowing
     - [ ] Respect structural boundaries
     - [ ] Maintain context across windows

2. Create DataLoader:
   - [ ] Define batch creation logic
   - [ ] Implement padding for batches
   - [ ] Add data augmentation if needed

## 🤖 Testing Tasks
1. Create unit tests:
   - [x] Test data loading
   - [x] Test preprocessing
   - [x] Test vocabulary creation
   - [-] Test sequence generation
     - [x] Basic sequence creation
     - [ ] Test boundary handling
     - [ ] Test context preservation
   - [ ] Test French-specific characters
   - [ ] Test proper tag nesting
     - [ ] Validate CHAR within SCENE
     - [ ] Validate SCENE within ACT
     - [ ] Validate LINE within CHAR

## 📚 Documentation Tasks
1. Create documentation:
   - [ ] Add docstrings
   - [ ] Create README
   - [ ] Document model architecture
   - [ ] Add usage examples 