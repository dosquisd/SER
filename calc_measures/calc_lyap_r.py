import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

sep = os.path.sep
current_dir = os.path.dirname(os.path.abspath(__file__))
dir = sep.join(current_dir.split(sep)[:-1])
sys.path.append(dir)

from my_functions import calc_lyap_r


load_dotenv()

# Obtener las variables básicas para la busqueda del coeficiente de lyapunov
records_df: pd.DataFrame = pd.read_csv(fr'{dir}\data.csv', delimiter=';', index_col=0)

partition: int = int(os.getenv('partition'))  # La partición que se utilizará
mili_s: float = float(os.getenv('mili_s'))  # Variable para indicar el periodo con el que se interpolará
mode: str = os.getenv('mode')  # Modo de interpolación
emb_dim: int = int(os.getenv('emb_dim'))  # Embedding dimension
orig: bool = False  # Variable para indicar que no tomaremos los datos originales, sino un resampleo

# Comenzar a calcular el coeficiente de lyapunov y guardarla en un csv
files: np.ndarray = records_df['File'].to_numpy()
new_label: str = 'Lyapunov Exponent'
out_path: str = f'{dir}\metrics\lyap_r{"" if orig else "1"}.csv'

try:
	lyap_rs, srs = calc_lyap_r(files, partition, emb_dim, orig, mili_s, mode)
	records_df[new_label] = lyap_rs
	records_df['Sample Rate'] = srs
except Exception as e:
	print(repr(e))
finally:
	records_df.to_csv(out_path, sep=';')
