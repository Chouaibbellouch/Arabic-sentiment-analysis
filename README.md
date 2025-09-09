# Arabic Sentiment Analysis

Un projet d'analyse de sentiment pour le texte arabe utilisant différents modèles de deep learning (NBoW, LSTM, GRU).

## 🚀 Installation

1. Clonez le repository :
```bash
git clone https://github.com/Chouaibbellouch/Arabic-sentiment-analysis.git
cd Arabic-sentiment-analysis
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Dataset

Le projet utilise le dataset `arbml/Arabic_Sentiment_Twitter_Corpus` de Hugging Face, contenant des tweets arabes avec des labels de sentiment (positif/négatif).

## 🏃‍♂️ Entraînement

Entraînez un modèle avec la commande suivante :

```bash
# Modèle NBoW (Neural Bag of Words)
python main.py nbow

# Modèle LSTM
python main.py lstm

# Modèle GRU
python main.py gru
```

Les modèles entraînés sont sauvegardés dans le dossier `checkpoints/` avec les noms :
- `nbow.pt`
- `lstm.pt` 
- `gru.pt`

## 🔮 Prédiction

Utilisez un modèle entraîné pour prédire le sentiment d'un texte arabe :

```bash
# Prédiction simple (NBoW par défaut)
python predict.py "أحب هذا المنتج كثيراً"

# Prédiction avec LSTM
python predict.py "هذا المنتج سيء جداً" --model-type lstm

# Prédiction avec GRU
python predict.py "لا أعرف رأيي في هذا" --model-type gru

# Prédiction avec affichage détaillé
python predict.py "النص العربي" --model-type lstm --verbose

# Prédiction avec un modèle spécifique
python predict.py "النص العربي" --model-type nbow --model checkpoints/nbow.pt
```

## ⚙️ Configuration

Les hyperparamètres peuvent être modifiés dans `config.py` :

- `EMBEDDING_DIM` : Dimension des embeddings (défaut: 128)
- `HIDDEN_DIM` : Dimension cachée pour LSTM/GRU (défaut: 256)
- `N_LAYERS` : Nombre de couches pour LSTM/GRU (défaut: 2)
- `DROPOUT` : Taux de dropout (défaut: 0.5)
- `BATCH_SIZE` : Taille des batches (défaut: 512)
- `NUM_EPOCHS` : Nombre d'époques (défaut: 10)
- `LR` : Learning rate (automatique selon le modèle)


