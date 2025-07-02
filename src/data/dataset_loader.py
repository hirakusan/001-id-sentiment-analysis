import pandas as pd
from datasets import load_dataset
import os

class HuggingFaceDatasetLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.df = None

    def load_dataset(self):
        dataset = load_dataset(self.dataset_name, split="train")
        self.df = dataset.to_pandas()
        return self.df

    def explore_dataset(self):
        if self.df is None:
            raise ValueError("Dataset belum di-load. Jalankan load_dataset() terlebih dahulu.")

        info = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.to_dict(),
            'null_values': self.df.isnull().sum().to_dict(),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        return info

    def prepare_data(self, text_column, label_column, test_size=0.2, stratify=True, random_state=42):
        from sklearn.model_selection import train_test_split
        df = self.df.dropna(subset=[text_column, label_column])
        stratify_col = df[label_column] if stratify else None
        return train_test_split(df, test_size=test_size, stratify=stratify_col, random_state=random_state)

    def save_processed_data(self, train_df, test_df):
        train_path = "data/processed/train.csv"
        test_path = "data/processed/test.csv"
        os.makedirs("data/processed", exist_ok=True)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        return train_path, test_path
