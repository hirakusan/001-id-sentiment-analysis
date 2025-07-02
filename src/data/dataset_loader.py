import pandas as pd
import numpy as np
from datasets import load_dataset
import logging
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
import os

class HuggingFaceDatasetLoader:
    """
    Loader untuk dataset HuggingFace adealvii/Cleaned-Indonesian-Tweet
    """
    
    def __init__(self, dataset_name: str = "adealvii/Cleaned-Indonesian-Tweet"):
        self.dataset_name = dataset_name
        self.dataset = None
        self.df = None
        
    def load_dataset(self, cache_dir: str = "./data/raw") -> pd.DataFrame:
        """
        Load dataset dari HuggingFace
        
        Returns:
            pandas.DataFrame: Dataset yang sudah di-load
        """
        try:
            logging.info(f"Loading dataset: {self.dataset_name}")
            
            # Load dataset dari HuggingFace
            self.dataset = load_dataset(
                self.dataset_name, 
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            # Convert ke pandas DataFrame
            if 'train' in self.dataset:
                self.df = self.dataset['train'].to_pandas()
            else:
                # Jika tidak ada split, ambil yang pertama
                first_split = list(self.dataset.keys())[0]
                self.df = self.dataset[first_split].to_pandas()
            
            logging.info(f"Dataset loaded successfully. Shape: {self.df.shape}")
            logging.info(f"Columns: {self.df.columns.tolist()}")
            
            return self.df
            
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            return None
    
    def explore_dataset(self) -> Dict[str, Any]:
        """
        Eksplorasi awal dataset
        
        Returns:
            Dict dengan informasi dataset
        """
        if self.df is None:
            raise ValueError("Dataset belum di-load. Jalankan load_dataset() terlebih dahulu.")
        
        info = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.to_dict(),
            'null_values': self.df.isnull().sum().to_dict(),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        # Jika ada kolom sentiment/label
        label_columns = [col for col in self.df.columns if 'label' in col.lower() or 'sentiment' in col.lower()]
        if label_columns:
            for col in label_columns:
                info[f'{col}_distribution'] = self.df[col].value_counts().to_dict()
        
        # Text column analysis
        text_columns = [col for col in self.df.columns if 'text' in col.lower() or 'tweet' in col.lower()]
        if text_columns:
            text_col = text_columns[0]
            info['text_stats'] = {
                'avg_length': self.df[text_col].str.len().mean(),
                'min_length': self.df[text_col].str.len().min(),
                'max_length': self.df[text_col].str.len().max(),
                'median_length': self.df[text_col].str.len().median()
            }
        
        return info
    
    def prepare_data(self, 
                    text_column: str = None, 
                    label_column: str = None,
                    test_size: float = 0.2,
                    stratify: bool = True,
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data untuk training
        
        Args:
            text_column: Nama kolom text (auto-detect jika None)
            label_column: Nama kolom label (auto-detect jika None)
            test_size: Proporsi data test
            stratify: Apakah menggunakan stratified split
            random_state: Random seed
            
        Returns:
            Tuple[train_df, test_df]
        """
        if self.df is None:
            raise ValueError("Dataset belum di-load.")
        
        # Auto-detect columns jika tidak dispesifikasi
        if text_column is None:
            text_columns = [col for col in self.df.columns if 'text' in col.lower() or 'tweet' in col.lower()]
            text_column = text_columns[0] if text_columns else self.df.columns[0]
        
        if label_column is None:
            label_columns = [col for col in self.df.columns if 'label' in col.lower() or 'sentiment' in col.lower()]
            label_column = label_columns[0] if label_columns else self.df.columns[1]
        
        logging.info(f"Using text column: {text_column}")
        logging.info(f"Using label column: {label_column}")
        
        # Clean data
        df_clean = self.df.dropna(subset=[text_column, label_column]).copy()
        df_clean = df_clean[df_clean[text_column].str.len() > 0]
        
        # Split data
        stratify_column = df_clean[label_column] if stratify else None
        
        train_df, test_df = train_test_split(
            df_clean,
            test_size=test_size,
            stratify=stratify_column,
            random_state=random_state
        )
        
        logging.info(f"Data split - Train: {len(train_df)}, Test: {len(test_df)}")
        
        return train_df, test_df
    
    def save_processed_data(self, 
                           train_df: pd.DataFrame, 
                           test_df: pd.DataFrame,
                           save_dir: str = "data/processed"):
        """
        Save processed data
        """
        os.makedirs(save_dir, exist_ok=True)
        
        train_path = os.path.join(save_dir, "train_data.csv")
        test_path = os.path.join(save_dir, "test_data.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logging.info(f"Data saved to {train_path} and {test_path}")
        
        return train_path, test_path

# Utility function untuk quick load
def load_indonesian_tweet_dataset() -> pd.DataFrame:
    """Quick loader function"""
    loader = HuggingFaceDatasetLoader()
    return loader.load_dataset()
    
def get_dataset_info() -> Dict[str, Any]:
    """Quick dataset info function"""
    loader = HuggingFaceDatasetLoader()
    loader.load_dataset()
    return loader.explore_dataset()