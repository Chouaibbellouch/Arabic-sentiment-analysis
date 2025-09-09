import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from dataset import Dataset
from vocab import Vocab
from dataloaders import create_data_loaders
from model import NBoW
from train import train
from evaluate import evaluate
from utils import set_seed, save_checkpoint

def main():

    config = Config()
    set_seed(config.SEED)
    
    print("Loading dataset...")
    dataset = Dataset(config)
    
    print("Creating vocabulary...")
    vocab = Vocab(dataset.get_train(), min_freq=config.VOCAB_MIN_FREQ)
    print(f"Vocabulary size: {len(vocab)}")
    
    print("Numericalizing data...")
    dataset.apply_numericalization(vocab)
    
    print("Creating data loaders...")
    data_loaders = create_data_loaders(
        dataset.get_train(),
        dataset.get_validation(), 
        dataset.get_test(),
        config.BATCH_SIZE,
        vocab.get_pad_index()
    )
    
    print("Creating model...")
    model = NBoW(
        vocab_size=len(vocab),
        embedding_dim=config.EMBEDDING_DIM,
        output_dim=config.OUTPUT_DIM,
        pad_index=vocab.get_pad_index()
    )  

    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        train_loss, train_acc = train(data_loaders["train"], model, criterion, optimizer)
        
        val_loss, val_acc = evaluate(data_loaders["validation"], model, criterion)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'config': config,
                'val_acc': val_acc,
                'epoch': epoch
            }, config.SAVE_DIR, config.MODEL_NAME)
    
    print("\nEvaluating on test set...")
    test_loss, test_acc = evaluate(data_loaders["test"], model, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
if __name__ == "__main__":
    main()