# Arabic Sentiment Analysis - NBoW

A simple PyTorch implementation for Arabic sentiment analysis using Neural Bag of Words.

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train the model:**
```bash
python main.py
```

3. **Predict sentiment:**
```bash
python predict.py "شكراً لكل من سأل عني، أنتم نعمة 💕"
# Output: pos (0.99)
```

## Usage

### Basic prediction:
```bash
python predict.py "النص العربي هنا"
```

### Detailed prediction:
```bash
python predict.py "النص العربي هنا" --verbose
```

## Files

- `main.py` - Train the model
- `predict.py` - Predict sentiment from command line
- `config.py` - Model configuration
- `model.py` - NBoW model implementation
- `dataset.py` - Data loading and preprocessing
- `vocab.py` - Vocabulary management

## Requirements

- PyTorch
- datasets
- NLTK
- NumPy
- tqdm

## Dataset

Uses the Arabic Sentiment Twitter Corpus with 47,000 training tweets.

## Results

Typical performance: ~94% accuracy on test set.