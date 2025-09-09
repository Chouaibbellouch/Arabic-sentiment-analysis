import argparse
import torch.nn as nn
import torch.optim as optim

from config import Config
from dataset import Dataset
from vocab import Vocab
from utils import set_seed, save_checkpoint, get_data_loader, build_model, initialize_weights
from train import train
from evaluate import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", choices=["nbow", "lstm", "gru"], help="Type de modèle à entraîner")
    args = parser.parse_args()

    config = Config(model_type=args.model_type)
    set_seed(config.SEED)

    print("Loading dataset...")
    dataset = Dataset(config)

    print("Creating vocabulary...")
    vocab = Vocab(dataset.get_train(), min_freq=config.VOCAB_MIN_FREQ)
    print(f"Vocabulary size: {len(vocab)}")

    print("Numericalizing data...")
    dataset.apply_numericalization(vocab)

    print("Creating data loaders...")
    train_dataloader = get_data_loader(dataset.get_train(), config.BATCH_SIZE, vocab.get_pad_index(), shuffle=True)
    valid_dataloader = get_data_loader(dataset.get_validation(), config.BATCH_SIZE, vocab.get_pad_index())
    test_dataloader = get_data_loader(dataset.get_test(), config.BATCH_SIZE, vocab.get_pad_index())

    print("Creating model...")
    model, mt = build_model(args.model_type, len(vocab), vocab.get_pad_index(), config)

    if mt == 'lstm' or mt == 'gru':
        model.apply(initialize_weights)
        
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    best_val_acc = 0.0

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")

        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, model_type=mt)
        val_loss, val_acc = evaluate(valid_dataloader, model, criterion, model_type=mt)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "config": config,
                "val_acc": val_acc,
                "epoch": epoch,
            }, config.SAVE_DIR, config.MODEL_NAME)

    print("\nEvaluating on test set...")
    test_loss, test_acc = evaluate(test_dataloader, model, criterion, model_type=mt)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()