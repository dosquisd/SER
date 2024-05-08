import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

sep = os.path.sep
current_dir = os.path.dirname(os.path.abspath(__file__))
dir = sep.join(current_dir.split(sep)[:-1])
sys.path.append(dir)

from my_functions import calc_lyap_e


load_dotenv()

# Obtener las variables básicas para la busqueda de los coeficientes de lyapunov
records_df: pd.DataFrame = pd.read_csv(fr'{dir}\data.csv', delimiter=';', index_col=0)

partition: int = int(os.getenv('partition'))  # La partición que se utilizará
mili_s: float = float(os.getenv('mili_s'))  # Variable para indicar el periodo con el que se interpolará
mode: str = os.getenv('mode')  # Modo de interpolación
orig: bool = False  # Variable para indicar que no tomaremos los datos originales, sino un resampleo

# Comenzar a calcular los coeficientes de lyapunov y guardarla en un csv
files: np.ndarray = records_df['File'].to_numpy()
new_label: str = 'Lyapunov Exponent'
out_path: str = f'{dir}\metrics\lyap_e{"" if orig else "1"}.csv'

try:
	lyap_es, srs = calc_lyap_e(files, partition, orig, mili_s, mode)
	records_df[f'1st {new_label}'] = lyap_es[:, 0]
	records_df[f'2nd {new_label}'] = lyap_es[:, 1]
	records_df[f'3rd {new_label}'] = lyap_es[:, 2]
	records_df[f'4th {new_label}'] = lyap_es[:, 3]
	records_df['Sample Rate'] = srs
except Exception as e:
	pass
finally:
	records_df.to_csv(out_path, sep=';')
