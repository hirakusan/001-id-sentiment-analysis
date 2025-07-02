import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class DatasetConfig:
    """Dataset Configuration"""
    name: str = "adealvii/Cleaned-Indonesian-Tweet"
    cache_dir: str = "./data/raw"
    text_column: str = "text"  # Will be auto-detected
    label_column: str = "label"  # Will be auto-detected
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    
@dataclass 
class ModelConfig:
    """Model Configuration"""
    model_name: str = 'indonesian-sentiment-classifier'
    algorithms: List[str] = field(default_factory=lambda: [
        'naive_bayes', 'logistic_regression', 'svm', 
        'random_forest', 'lstm', 'bert'
    ])
    
    # Feature extraction settings
    max_features: int = 10000
    max_len: int = 128
    ngram_range: tuple = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    
    # Neural network settings
    embedding_dim: int = 100
    lstm_units: int = 64
    dropout_rate: float = 0.3
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    
@dataclass
class PreprocessConfig:
    """Text Preprocessing Configuration"""
    lowercase: bool = True
    remove_urls: bool = True
    remove_mentions: bool = False
    remove_hashtags: bool = False
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_extra_whitespace: bool = True
    normalize_slang: bool = True
    remove_stopwords: bool = True
    apply_stemming: bool = False
    min_text_length: int = 3
    
@dataclass
class AppConfig:
    """Application Configuration"""
    app_name: str = "Indonesian Sentiment Analysis"
    version: str = "1.0.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
class Config:
    """Main Configuration Class"""
    
    def __init__(self):
        self.dataset = DatasetConfig()
        self.model = ModelConfig()
        self.preprocess = PreprocessConfig()
        self.app = AppConfig()
        
    # Indonesian stopwords (expanded)
    INDONESIAN_STOPWORDS = {
        'yang', 'ini', 'itu', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada',
        'adalah', 'dalam', 'tidak', 'akan', 'atau', 'juga', 'telah', 'dapat', 'sudah',
        'harus', 'bisa', 'masih', 'saya', 'kamu', 'dia', 'mereka', 'kita', 'kami',
        'ada', 'jadi', 'karena', 'kalau', 'tapi', 'lalu', 'maka', 'bila', 'sambil',
        'setelah', 'sebelum', 'selama', 'hingga', 'sampai', 'sejak', 'tanpa', 'kecuali',
        'antara', 'oleh', 'bagi', 'menurut', 'terhadap', 'atas', 'bawah', 'depan',
        'belakang', 'samping', 'tengah', 'luar', 'sekitar', 'dekat', 'jauh'
    }
    
    # Indonesian slang dictionary (comprehensive)
    SLANG_DICT = {
        # Pronouns
        'gw': 'saya', 'gue': 'saya', 'w': 'saya', 'lu': 'kamu', 'lo': 'kamu',
        'elu': 'kamu', 'dia': 'dia', 'mrk': 'mereka',
        
        # Common words
        'banget': 'sangat', 'bgt': 'sangat', 'bgd': 'sangat', 'bener': 'benar',
        'bnar': 'benar', 'udah': 'sudah', 'udh': 'sudah', 'dah': 'sudah',
        'blm': 'belum', 'blom': 'belum', 'ga': 'tidak', 'gak': 'tidak',
        'kagak': 'tidak', 'kaga': 'tidak', 'engga': 'tidak', 'enggak': 'tidak',
        'skrg': 'sekarang', 'skg': 'sekarang', 'skrang': 'sekarang',
        'ntar': 'nanti', 'tar': 'nanti', 'emg': 'memang', 'emang': 'memang',
        
        # Time expressions
        'kmrn': 'kemarin', 'kmaren': 'kemarin', 'bsk': 'besok', 'besok': 'besok',
        'td': 'tadi', 'pagi': 'pagi', 'malem': 'malam', 'mlm': 'malam',
        
        # Expressions
        'tp': 'tapi', 'trs': 'terus', 'trus': 'terus', 'klo': 'kalau',
        'kalo': 'kalau', 'krn': 'karena', 'krna': 'karena', 'jd': 'jadi',
        'jadi': 'jadi', 'yg': 'yang', 'dg': 'dengan', 'dgn': 'dengan',
        
        # Internet slang
        'wkwk': '', 'wkwkwk': '', 'hehe': '', 'hihi': '', 'haha': '',
        'wkwkwkwk': '', 'kwkwkw': '', 'awkwkwk': '', 'xixixi': '',
        'kekw': '', 'lol': '', 'lmao': '',
        
        # Abbreviations
        'org': 'orang', 'orng': 'orang', 'tmp': 'tempat', 'tmpt': 'tempat',
        'krj': 'kerja', 'krja': 'kerja', 'skl': 'sekolah', 'sklh': 'sekolah',
        'rmh': 'rumah', 'byk': 'banyak', 'bnyk': 'banyak', 'sdkt': 'sedikit',
        'sdikit': 'sedikit'
    }
    
    # Sentiment mappings
    SENTIMENT_LABELS = {
        'positive': 2, 'neutral': 1, 'negative': 0,
        'pos': 2, 'neu': 1, 'neg': 0,
        2: 'positive', 1: 'neutral', 0: 'negative'
    }
    
    # Emotion labels (if available)
    EMOTION_LABELS = {
        'anger': 0, 'fear': 1, 'joy': 2, 'sadness': 3, 'surprise': 4,
        0: 'anger', 1: 'fear', 2: 'joy', 3: 'sadness', 4: 'surprise'
    }