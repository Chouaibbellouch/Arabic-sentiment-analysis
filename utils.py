import os
import torch
import torch.nn as nn
import numpy as np

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_checkpoint(state: dict, folder_path: str, file_name: str):
    ensure_dir(folder_path)
    torch.save(state, os.path.join(folder_path, file_name))

def load_checkpoint(model_path: str):
    return torch.load(model_path)

def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_length = [i["length"] for i in batch]
        batch_length = torch.stack(batch_length)
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "length": batch_length, "label": batch_label}
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy

def build_model(model_type, vocab_size, pad_index, config):
    model_type = model_type.lower()
    if model_type == "nbow":
        from models.nbow import NBoW
        return NBoW(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            output_dim=config.OUTPUT_DIM,
            pad_index=pad_index,
        ), "nbow"
    elif model_type == "lstm":
        from models.lstm import LSTM
        return LSTM(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            n_layers=config.N_LAYERS,
            dropout_rate=config.DROPOUT,
            pad_index=pad_index,
        ), "lstm"
    elif model_type == "gru":
        from models.gru import GRU
        return GRU(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            n_layers=config.N_LAYERS,
            dropout_rate=config.DROPOUT,
            pad_index=pad_index,
        ), "gru"
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'nbow', 'lstm' or 'gru'.")

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.orthogonal_(param)