import argparse
import torch
import nltk
import sys
import os
from utils import load_checkpoint, build_model
from config import Config

def predict_text(text, model_type="nbow", model_path=None):
    config = Config(model_type=model_type)

    if model_path is None:
        model_path = os.path.join(config.SAVE_DIR, config.MODEL_NAME)

    if not os.path.exists(model_path):
        print(f"Erreur: Le modèle {model_path} n'existe pas.")
        print("Assurez-vous d'avoir entraîné le modèle avec 'python main.py <model_type>'")
        return None, None, None, None

    try:
        checkpoint = load_checkpoint(model_path)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None, None, None, None

    vocab = checkpoint["vocab"]
    model_config = checkpoint["config"]

    vocab_size = len(vocab) if hasattr(vocab, "__len__") else len(vocab.itos)
    pad_index = vocab.get_pad_index()

    model, mt = build_model(model_type, vocab_size, pad_index, model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    try:
        tokens = nltk.word_tokenize(text)
    except:
        print("Erreur: NLTK punkt_tab n'est pas téléchargé. Exécutez d'abord le script principal.")
        return None, None, None, None

    ids = vocab.lookup_indices(tokens)
    input_ids = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        if mt in ["lstm", "gru"]:
            length = torch.tensor([len(ids)], dtype=torch.long)
            logits = model(input_ids, length).squeeze(0)
        else:
            logits = model(input_ids).squeeze(0)
        probability = torch.softmax(logits, dim=-1)
        predicted_class = logits.argmax(dim=-1).item()
        predicted_probability = probability[predicted_class].item()

    sentiment_labels = ["neg", "pos"]
    sentiment = sentiment_labels[predicted_class]

    return predicted_class, predicted_probability, sentiment, probability

def main():
    parser = argparse.ArgumentParser(description="Prédiction de sentiment en arabe")
    parser.add_argument("text", help="Texte arabe à analyser")
    parser.add_argument("--model-type", "-t", choices=["nbow", "lstm", "gru"], default="nbow", help="Type de modèle")
    parser.add_argument("--model", "-m", help="Chemin vers le modèle (optionnel)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Affichage détaillé")

    args = parser.parse_args()

    predicted_class, predicted_probability, sentiment, probabilities = predict_text(args.text, args.model_type, args.model)

    if predicted_class is not None:
        if args.verbose:
            print(f"Texte: {args.text}")
            print(f"Sentiment: {sentiment}")
            print(f"Confiance: {predicted_probability:.4f}")
            print(f"Probabilité neg: {probabilities[0].item():.4f}")
            print(f"Probabilité pos: {probabilities[1].item():.4f}")
        else:
            print(f"{sentiment} ({predicted_probability:.2f})")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()