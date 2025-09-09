from collections import Counter, defaultdict

class Vocab:
    def __init__(self, train_data, min_freq=5):
        counter = Counter(token for tokens in train_data["tokens"] for token in tokens)
        
        self.special_tokens = ["<unk>", "<pad>"]
        
        vocab_tokens = [t for t, c in counter.items() if c >= min_freq]
        self.stoi = {tok: i for i, tok in enumerate(self.special_tokens + vocab_tokens)}
        self.itos = {i: tok for tok, i in self.stoi.items()}
        
        self.unk_index = self.stoi["<unk>"]
        self.pad_index = self.stoi["<pad>"]
        
        # Ne pas utiliser defaultdict avec lambda pour la sérialisation
        self._stoi_dict = dict(self.stoi)  # Garder une copie du dictionnaire original
    
    def lookup_indices(self, tokens):
        return [self.stoi.get(token, self.unk_index) for token in tokens]
    
    def lookup_tokens(self, ids):
        return [self.itos[Id] for Id in ids]
    
    def __len__(self):
        return len(self.itos)
    
    def get_unk_index(self):
        return self.unk_index
    
    def get_pad_index(self):
        return self.pad_index
    
    def __getstate__(self):
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.stoi = defaultdict(lambda: self.unk_index, self._stoi_dict)