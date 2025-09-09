import torch
import tqdm
import numpy as np
from utils import get_accuracy


def evaluate(dataloader, model, criterion, model_type = 'nbow'):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="evaluating..."):
            ids = batch["ids"]
            length = batch["length"]
            label = batch["label"]
            prediction = model(ids, length) if model_type != 'nbow' else model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)