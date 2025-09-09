import datasets
import nltk
from collections import Counter
from datasets import Features, Value, ClassLabel, Sequence

class Dataset:
    def __init__(self, config):

        nltk.download("punkt_tab")

        self.ds = datasets.load_dataset('arbml/Arabic_Sentiment_Twitter_Corpus')
        
        train_val_test = self.ds['test'].train_test_split(test_size=0.5, seed=42)
        
        self.ds = {
            "train": self.ds["train"],
            "validation": train_val_test["train"],
            "test": train_val_test["test"],
        }
        
        self.target_features = Features({
            "tweet": Value("string"),
            "label": ClassLabel(names=["neg", "pos"]),
            "tokens": Sequence(Value("string"))
        })
        
        self._prepare_dataset()
    
    def tokenize(self, example, tokenizer, max_length):
        """Fonction de tokenisation pour map"""
        tokens = tokenizer(example["tweet"])[:max_length]
        return {"tokens": tokens}
    
    def _prepare_dataset(self):
        """Prépare le dataset avec tokenisation et features"""
        max_length = 256
        
        self.ds['train'] = self.ds['train'].map(
            self.tokenize, 
            fn_kwargs={"tokenizer": nltk.word_tokenize, "max_length": max_length}, 
            features=self.target_features
        )
        
        self.ds['validation'] = self.ds['validation'].map(
            self.tokenize, 
            fn_kwargs={"tokenizer": nltk.word_tokenize, "max_length": max_length}, 
            features=self.target_features
        )
        
        self.ds['test'] = self.ds['test'].map(
            self.tokenize, 
            fn_kwargs={"tokenizer": nltk.word_tokenize, "max_length": max_length}, 
            features=self.target_features
        )

    def numericalize_example(self, example):
        ids = self.lookup_indices(example["tokens"])
        return {"ids": ids}
    
    def apply_numericalization(self, vocab):
        self.lookup_indices = vocab.lookup_indices
        
        self.ds['train'] = self.ds['train'].map(self.numericalize_example)
        self.ds['validation'] = self.ds['validation'].map(self.numericalize_example)
        self.ds['test'] = self.ds['test'].map(self.numericalize_example)
        
        self.ds['train'] = self.ds['train'].with_format(type="torch", columns=["ids", "label"])
        self.ds['validation'] = self.ds['validation'].with_format(type="torch", columns=["ids", "label"])
        self.ds['test'] = self.ds['test'].with_format(type="torch", columns=["ids", "label"])
    
    def get_train(self):
        return self.ds["train"]
    
    def get_validation(self):
        return self.ds["validation"]
    
    def get_test(self):
        return self.ds["test"]