import argparse
import torch
import nltk
import sys
import os
from model import NBoW
from utils import load_checkpoint
from config import Config

def predict_text(text, model_path=None):

    config = Config()
    
    if model_path is None:
        model_path = os.path.join(config.SAVE_DIR, config.MODEL_NAME)
    
    if not os.path.exists(model_path):
        print(f"Erreur: Le modèle {model_path} n'existe pas.")
        print("Assurez-vous d'avoir entraîné le modèle avec 'python main.py'")
        return None, None, None
    
    try:
        checkpoint = load_checkpoint(model_path)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None, None, None
    
    vocab = checkpoint['vocab']
    model_config = checkpoint['config']
    
    model = NBoW(
        vocab_size=len(vocab.itos),
        embedding_dim=model_config.EMBEDDING_DIM,
        output_dim=model_config.OUTPUT_DIM,
        pad_index=vocab.get_pad_index()
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    try:
        tokens = nltk.word_tokenize(text)
    except:
        print("Erreur: NLTK punkt_tab n'est pas téléchargé. Exécutez d'abord le script principal.")
        return None, None, None
    
    ids = vocab.lookup_indices(tokens)
    
    input_tensor = torch.tensor([ids], dtype=torch.long)
    
    with torch.no_grad():
        prediction = model(input_tensor).squeeze(dim=0)
        probability = torch.softmax(prediction, dim=-1)
        predicted_class = prediction.argmax(dim=-1).item()
        predicted_probability = probability[predicted_class].item()
    
    sentiment_labels = ["neg", "pos"]
    sentiment = sentiment_labels[predicted_class]
    
    return predicted_class, predicted_probability, sentiment, probability

def main():
    parser = argparse.ArgumentParser(description='Prédiction de sentiment en arabe')
    parser.add_argument('text', help='Texte arabe à analyser')
    parser.add_argument('--model', '-m', help='Chemin vers le modèle (optionnel)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Affichage détaillé')
    
    args = parser.parse_args()
    
    predicted_class, predicted_probability, sentiment, probabilities = predict_text(args.text, args.model)
    
    if predicted_class is not None:
        if args.verbose:
            print(f"Texte: {args.text}")
            print(f"Sentiment: {sentiment}")
            print(f"Confiance: {predicted_probability:.4f}")
            print(f"Probabilité neg: {probabilities[0].item():.4f}")  # Index 0 = neg
            print(f"Probabilité pos: {probabilities[1].item():.4f}")  # Index 1 = pos
        else:
            print(f"{sentiment} ({predicted_probability:.2f})")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()