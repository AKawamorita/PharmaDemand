from pathlib import Path
import pandas as pd
import os
import glob
from datetime import datetime

class CSVLoader:
    # Metodos protegidos 
    def _extract_features_from_csvfile(self, file_path, csv_file):
        df = pd.read_csv(file_path)
        return df
    
    def create_dataframe(self, file_path, csv_file):
        arquivo = Path(file_path / csv_file) 

        df = pd.read_csv(arquivo)
        return df