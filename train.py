import tqdm
import numpy as np
from utils import get_accuracy

def train(dataloader, model, criterion, optimizer, model_type = 'nbow'):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(dataloader, desc="training..."):
        ids = batch["ids"]
        length = batch["length"]
        label = batch["label"]
        prediction = model(ids, length) if model_type != 'nbow' else model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)