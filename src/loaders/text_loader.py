from pathlib import Path
import pandas as pd
import os
import glob
from datetime import datetime
from src import features as ft

class TextLoader:
    # Metodos protegidos 
    def _extract_features_from_file(self, file_path, run_id, time_index, max_sensors=4):
        df = self.read_bearing_file(file_path)
        row_features = {
            "run_id": run_id,
            "time_index": time_index,
            "file_name": os.path.basename(file_path)
        }

        n_sensors = min(df.shape[1], max_sensors)

        for sensor_idx in range(n_sensors):
            signal = df.iloc[:, sensor_idx].values
            sensor_prefix = f"s{sensor_idx+1}_"
            sensor_feats = ft.extract_signal_features(signal, prefix=sensor_prefix)
            row_features.update(sensor_feats)

        return row_features

    # Metodos publicos
    def read_bearing_file(self, file_path):
        df = pd.read_csv(file_path, sep=r"\s+", header=None)
        return df

   
    def create_dataframe(self, run_folder, run_id, max_sensors=4):
        files = sorted(glob.glob(os.path.join(run_folder, "*")))
        print(f"\nRun: {run_id}")
        print(f"Pasta: {run_folder}")
        print(f"Qtd. arquivos encontrados: {len(files)}")

        rows = []
        for time_index, file_path in enumerate(files):

            # EXTRAÇÃO DA DATA: Pega o nome do arquivo (ex: 2003.10.22.12.06.24)
            file_name = os.path.basename(file_path)
            try:
                # Converte o nome do arquivo em um objeto datetime real
                timestamp = datetime.strptime(file_name, '%Y.%m.%d.%H.%M.%S')
            except ValueError:
                # Caso o nome do arquivo não seja uma data (ex: .DS_Store ou logs)
                timestamp = None

            row = self._extract_features_from_file(
                file_path=file_path,
                run_id=run_id,
                time_index=time_index,
                max_sensors=max_sensors
            )
            # Adiciona a data extraída ao dicionário da linha
            row['timestamp'] = timestamp
            rows.append(row)

        print(f"Qtd. linhas geradas: {len(rows)}")
        df_run = pd.DataFrame(rows)

        print("Colunas do df_run:", df_run.columns.tolist())

        # CONVERSÃO E INDICE: Essencial para a função de alerta
        if 'timestamp' in df_run.columns:
            df_run['timestamp'] = pd.to_datetime(df_run['timestamp'])
            df_run.set_index('timestamp', inplace=True)

        max_time = df_run["time_index"].max()
        df_run["RUL"] = max_time - df_run["time_index"]

        return df_run
    
    