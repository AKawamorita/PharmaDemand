from pathlib import Path
import pandas as pd
import pyarrow

class ParquetRepository:
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)

    def _get_full_path(self, relative_path: str) -> Path:
        return self.base_path / relative_path
    
    def set_base_path (self, base_path: str = "data"):
        self.base_path = Path(base_path)

    def save(
        self,
        df: pd.DataFrame,
        relative_path: str,
        compression: str = "snappy",
        index: bool = False,
        overwrite: bool = True
    ) -> None:
        full_path = self._get_full_path(relative_path)

        # cria diretórios automaticamente
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if full_path.exists() and not overwrite:
            raise FileExistsError(f"Arquivo já existe: {full_path}")

        df.to_parquet(
            full_path,
            compression=compression,
            index=index
        )

        print(f"[OK] Arquivo salvo em: {full_path}")

    def load(
        self,
        relative_path: str,
        columns: list = None
    ) -> pd.DataFrame:
        full_path = self._get_full_path(relative_path)

        if not full_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {full_path}")

        df = pd.read_parquet(full_path, columns=columns)

        print(f"[OK] Arquivo carregado: {full_path}")

        return df

    def exists(self, relative_path: str) -> bool:
        return self._get_full_path(relative_path).exists()

    def delete(self, relative_path: str) -> None:
        full_path = self._get_full_path(relative_path)

        if full_path.exists():
            full_path.unlink()
            print(f"[OK] Arquivo removido: {full_path}")
        else:
            print(f"[WARN] Arquivo não encontrado: {full_path}")