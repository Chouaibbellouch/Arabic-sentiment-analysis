from dataclasses import dataclass, field

@dataclass
class Config:
    model_type: str = "nbow"
    DATASET : str = "arbml/Arabic_Sentiment_Twitter_Corpus"
    TEXT_COL: str = "tweet"
    LABEL_COL: str = "label"
    VOCAB_MIN_FREQ: int = 5
    MAX_LENGTH: int = 256
    EMBEDDING_DIM: int = 128
    HIDDEN_DIM: int = 256
    N_LAYERS: int = 2
    DROPOUT: float = 0.5
    OUTPUT_DIM: int = 2
    LR: float = field(init=False)
    BATCH_SIZE: int = 512
    NUM_EPOCHS: int = 10
    SEED: int = 1306
    SAVE_DIR: str = "checkpoints"
    MODEL_NAME: str = field(init=False)

    def __post_init__(self):
        name_map = {
            "nbow": "nbow.pt",
            "lstm": "lstm.pt",
            "gru": "gru.pt",
        }
        self.MODEL_NAME = name_map.get(self.model_type.lower(), f"{self.model_type}.pt")

        lr_map = {
            "nbow": 1e-3,
            "lstm": 5e-4,
            "gru": 5e-4,
        }
        self.LR = lr_map.get(self.model_type.lower(), 1e-3)