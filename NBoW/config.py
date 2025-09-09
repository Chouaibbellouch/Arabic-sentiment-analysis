from dataclasses import dataclass

@dataclass
class Config:
    DATASET : str = "arbml/Arabic_Sentiment_Twitter_Corpus"
    TEXT_COL: str = "tweet"
    LABEL_COL: str = "label"
    VOCAB_MIN_FREQ: int = 5
    MAX_LENGTH: int = 256
    EMBEDDING_DIM: int = 128
    OUTPUT_DIM: int = 2
    LR: float = 1e-3
    BATCH_SIZE: int = 512
    NUM_EPOCHS: int = 10
    SEED: int = 1306
    SAVE_DIR: str = "checkpoints"
    MODEL_NAME: str = "nbow.pt"