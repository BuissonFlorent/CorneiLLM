# Project Questions Tracker

## ✓ Answered Questions
1. What is the size of your French theater corpus?
   - Answer: Single play (Melite) for now, with plans to add more plays
   - Status: Fully answered

2. What's the vocabulary size of your word embeddings?
   - Answer: ~3,000 words currently, expected to stay under 10,000
   - Status: Fully answered

3. What's the dimension size of your word embeddings?
   - Answer: Updated to 100 dimensions
   - Status: Fully answered

4. What specific architecture are you planning to use?
   - Answer: Progressive complexity approach:
     1. Start with basic RNN
     2. Experiment with deeper architectures
     3. Move to GRU/LSTM
     4. Finally explore Transformers
   - Status: Fully answered

## 🔄 Partially Answered/Follow-up Questions
5. Questions about current embeddings:
   - Are you using any special tokens (currently only <UNK>)
   - Status: Partially answered, implementation pending

6. Architecture follow-up questions:
   - What metrics will you use to compare architectures?
   - What will be your baseline performance measure?
   - How will you ensure fair comparison between models?

## ❓ Outstanding Questions
7. Do you need to handle any specific preprocessing for French text?
   - Handling of accents?
   - Special characters?
   - Capitalization rules? 