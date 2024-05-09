import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

sep = os.path.sep
current_dir = os.path.dirname(os.path.abspath(__file__))
dir = sep.join(current_dir.split(sep)[:-1])
sys.path.append(dir)

from my_functions import calc_dfa


load_dotenv()

# Obtener las variables básicas para la busqueda del dfa
records_df: pd.DataFrame = pd.read_csv(fr'{dir}\data.csv', delimiter=';', index_col=0)

partition: int = int(os.getenv('partition'))  # La partición que se utilizará
mili_s: float = float(os.getenv('mili_s'))  # Variable para indicar el periodo con el que se interpolará
mode: str = os.getenv('mode')  # Modo de interpolación
orig: bool = False  # Variable para indicar que no tomaremos los datos originales, sino un resampleo

# Comenzar a calcular el dfa y guardarla en un csv
files: np.ndarray = records_df['File'].to_numpy()
new_label: str = 'Detrented Fluctuation Analysis'
out_path: str = f'{dir}\metrics\dfa{"" if orig else "1"}.csv'

try:
	dfas, srs = calc_dfa(files, partition, orig, mili_s, mode)
	records_df[new_label] = dfas
	records_df['Sample Rate'] = srs
except Exception as e:
	pass
finally:
	records_df.to_csv(out_path, sep=';')
