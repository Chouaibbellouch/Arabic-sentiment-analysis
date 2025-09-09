# nbow/utils.py
import os
import torch
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

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy