import librosa
import nolds
import os
import numpy as np
import pandas as pd
from my_functions.my_ravdess_functions import get_feature_ravdess


ravdess_path: str = 'Ravdess/audio_speech_actors_01-24'
SR: int = 22050

def resample(sample: np.ndarray[np.float32], mili_s: float = 0.1, mode: str = 'median') -> tuple[np.ndarray[np.float32], float]:
	"""Se hace una interpolación con los datos contenidos en `sample`, el cual consiste en una pista \
	de audio con una frecuencia de muestreo de SR (22050 hz)
	
	# Parametros

	sample : numpy.ndarray[np.float32]
		Pista de audio a interpolar con una frecuencia de muestro constante de 22050
	mili_s : float
		Es el periodo con el que se realizará el resampleo
	mode : str
		Es el modo de interpolación que se utilizará para el resampleo. Puede ser únicamente "mean" o "median"

	# Retorno

	Se retorna una tupla con el arreglo resultante después del resampleo en la primera posición, y en la segunda \
	es la nueva frecuencia de muestreo
	"""
	mode = mode.lower()

	if mode not in ('mean', 'median'):
		return np.zeros(1, np.float32), 0.0
	
	n = len(sample)
	time_range = pd.date_range(start='2024-05-06', periods=n, freq=f'{1e6/SR:.3f}us')  # La elección de la fecha es irrelevante
	new_sample_df = pd.DataFrame({'Audio': sample}, index=time_range).resample(f'{mili_s:.3f}ms')
	n_new = len(new_sample_df)
	sr_new = n_new*SR / n  # Sale de: SR/n = sr_new/n_new => n_new*SR/n

	if mode == 'mean':
		return new_sample_df.mean()['Audio'].to_numpy(), sr_new
	return new_sample_df.median()['Audio'].to_numpy(), sr_new

def get_audio_record_ravdess(file: str, orig: bool = False, mili_s: float = 0.1, mode: str = 'median') -> tuple[np.ndarray, float]:
	"""Dado un nombre de archivo que contenga alguna pista de audio del dataset de Ravdess, retorna un arreglo junto a la frecuencia de muestreo
	
	# Parametros
	
	file : str
		Es el nombre del archivo del cual se quiere conseguir su pista de audio
	orig : bool
		Indica si el programa retorne los datos originales antes del resampleo. Por defecto `orig = False`
	mili_s : float
		Es el periodo con el que se realizará el resampleo
	mode : str
		Es el modo de interpolación que se utilizará para el resampleo. Puede ser únicamente "mean" o "median"

	# Retorno

	Retorna una tupla, donde la primera posición corresponde al arreglo que contiene la información del audio y la segunda contiene el sample rate respectivo
	"""
	file = os.path.basename(file)
	actor_j = get_feature_ravdess(file, 7)
	sample, sr = librosa.load(f'{ravdess_path}/Actor_{actor_j:02}/{file}')

	return (sample, sr) if orig else resample(sample, mili_s, mode)

def calc_hurst_rs(files: np.ndarray[str], partitions: int = 1, orig: bool = False, mili_s: float = 0.1, mode: str = 'median') -> tuple[np.ndarray[np.float32], float]:
	"""
	Calcula el exponente de hurst (con `nolds.hurst_rs()`) para una lista de nombres de archivos que contienen los audios de ravdess

	# Parametros

	files : np.ndarray[str]
		Es la lista de pistas de audio del cual se quiere conseguir sus pistas de audio
	partitions : int
		Representa la partición de audio con la que se trabajará. Es útil cambiarla cuando se quiera probar las funciones, sin tener que utilizar toda la pista de audio .\
		Por defecto `partitions = 1`
	orig : bool
		Indica si el programa retorne los datos originales antes del resampleo. Por defecto `orig = False`
	mili_s : float
		Es el periodo con el que se realizará el resampleo. Por defecto `mili_s = 0.1`
	mode : str
		Es el modo de interpolación que se utilizará para el resampleo. Puede ser únicamente "mean" o "median". Por defecto `mode = 'median'`

	# Retorno

	Retorna una tupla, donde la primera posición corresponde al arreglo que contiene la métrica con `nolds.hurst_rs()` y la segunda contiene el sample rate respectivo
	"""
	hurst_rs_list = np.empty(files.shape, dtype='float32')

	for i, file in enumerate(files):
		y, sr = get_audio_record_ravdess(file, orig, mili_s, mode)
		hurst_rs_list[i] = nolds.hurst_rs(y[:len(y)//partitions], fit='poly')

	return hurst_rs_list, sr

def calc_dfa(files: np.ndarray[str], partitions: int = 1, orig: bool = False, mili_s: float = 0.1, mode: str = 'median') -> tuple[np.ndarray[np.float32], float]:
	"""
	Calcula el "Detrented Fluctuation Analysis" (con `nolds.dfa()`) para una lista de nombres de archivos que contienen los audios de ravdess

	# Parametros

	files : np.ndarray[str]
		Es la lista de pistas de audio del cual se quiere conseguir sus pistas de audio
	partitions : int
		Representa la partición de audio con la que se trabajará. Es útil cambiarla cuando se quiera probar las funciones, sin tener que utilizar toda la pista de audio .\
		Por defecto `partitions = 1`
	orig : bool
		Indica si el programa retorne los datos originales antes del resampleo. Por defecto `orig = False`
	mili_s : float
		Es el periodo con el que se realizará el resampleo. Por defecto `mili_s = 0.1`
	mode : str
		Es el modo de interpolación que se utilizará para el resampleo. Puede ser únicamente "mean" o "median". Por defecto `mode = 'median'`

	# Retorno

	Retorna una tupla, donde la primera posición corresponde al arreglo que contiene la métrica con `nolds.dfa()` y la segunda contiene el sample rate respectivo
	"""
	dfa_list = np.empty(files.shape, dtype='float32')

	for i, file in enumerate(files):
		y, sr = get_audio_record_ravdess(file, orig, mili_s, mode)
		dfa_list[i] = nolds.dfa(y[:len(y)//partitions], fit_exp='poly')

	return dfa_list, sr

def calc_lyap_e(files: np.ndarray[str], partitions: int = 1, orig: bool = False, mili_s: float = 0.1, mode: str = 'median') -> tuple[np.ndarray[np.float32], float]:
	"""
	Calcula los exponentes de Lyapunov (con `nolds.lyap_e()`) para una lista de nombres de archivos que contienen los audios de ravdess

	# Parametros

	files : np.ndarray[str]
		Es la lista de pistas de audio del cual se quiere conseguir sus pistas de audio
	partitions : int
		Representa la partición de audio con la que se trabajará. Es útil cambiarla cuando se quiera probar las funciones, sin tener que utilizar toda la pista de audio .\
		Por defecto `partitions = 1`
	orig : bool
		Indica si el programa retorne los datos originales antes del resampleo. Por defecto `orig = False`
	mili_s : float
		Es el periodo con el que se realizará el resampleo. Por defecto `mili_s = 0.1`
	mode : str
		Es el modo de interpolación que se utilizará para el resampleo. Puede ser únicamente "mean" o "median". Por defecto `mode = 'median'`

	# Retorno
	
	Retorna una tupla, donde la primera posición corresponde un arreglo multidimensional que contiene la métrica con `nolds.lyap_e()` y la segunda contiene el sample rate respectivo
	"""
	lyap_e_list = np.empty(files.shape, dtype=np.ndarray)

	for i, file in enumerate(files):
		y, sr = get_audio_record_ravdess(file, orig, mili_s, mode)
		lyap_e_list[i] = nolds.lyap_e(y[:len(y)//partitions])

	return np.array([list(exponents) for exponents in lyap_e_list]), sr

def calc_lyap_r(files: np.ndarray[str], partitions: int = 1, orig: bool = False, mili_s: float = 0.1, mode: str = 'median') -> tuple[np.ndarray[np.float32], float]:
	"""
	Calcula el exponente de Lyapunov (con `nolds.lyap_r()`) para una lista de nombres de archivos que contienen los audios de ravdess

	# Parametros

	files : np.ndarray[str]
		Es la lista de pistas de audio del cual se quiere conseguir sus pistas de audio
	partitions : int
		Representa la partición de audio con la que se trabajará. Es útil cambiarla cuando se quiera probar las funciones, sin tener que utilizar toda la pista de audio .\
		Por defecto `partitions = 1`
	orig : bool
		Indica si el programa retorne los datos originales antes del resampleo. Por defecto `orig = False`
	mili_s : float
		Es el periodo con el que se realizará el resampleo. Por defecto `mili_s = 0.1`
	mode : str
		Es el modo de interpolación que se utilizará para el resampleo. Puede ser únicamente "mean" o "median". Por defecto `mode = 'median'`

	# Retorno

	Retorna una tupla, donde la primera posición corresponde el arreglo que contiene la métrica con `nolds.lyap_r()` y la segunda contiene el sample rate respectivo
	"""
	lyap_r_list = np.empty(files.shape, dtype='float32')

	for i, file in enumerate(files):
		y, sr = get_audio_record_ravdess(file, orig, mili_s, mode)
		lyap_r_list[i] = nolds.lyap_r(y[:len(y)//partitions], fit='poly')
	
	return lyap_r_list, sr

def calc_corr_dim(files: np.ndarray, partitions: int = 1, emb_dim: int = 14, orig: bool = False, mili_s: float = 0.1, mode: str = 'median') -> tuple[np.ndarray[np.float32], float]:
	"""
	Calcula los exponentes de Lyapunov (con `nolds.lyap_e()`) para una lista de nombres de archivos que contienen los audios de ravdess

	# Parametros

	files : np.ndarray[str]
		Es la lista de pistas de audio del cual se quiere conseguir sus pistas de audio
	partitions : int
		Representa la partición de audio con la que se trabajará. Es útil cambiarla cuando se quiera probar las funciones, sin tener que utilizar toda la pista de audio .\
		Por defecto `partitions = 1`
	emb_dim : int
		Es el embedding dimension que se utilizará para calcular la dimensión de correlación. Por defecto `emb_dim = 14`
	orig : bool
		Indica si el programa retorne los datos originales antes del resampleo. Por defecto `orig = False`
	mili_s : float
		Es el periodo con el que se realizará el resampleo. Por defecto `mili_s = 0.1`
	mode : str
		Es el modo de interpolación que se utilizará para el resampleo. Puede ser únicamente "mean" o "median". Por defecto `mode = 'median'`

	# Retorno
		
	Retorna una tupla, donde la primera posición corresponde un arreglo multidimensional que contiene la métrica con `nolds.corr_dim()` y la segunda contiene el sample rate respectivo
	"""

	corr_dim_list = np.empty(files.shape, dtype='float32')

	for i, file in enumerate(files):
		y, sr = get_audio_record_ravdess(file, orig, mili_s, mode)
		corr_dim_list[i] = nolds.corr_dim(y[:len(y)//partitions], emb_dim=emb_dim, fit='poly')
	
	return corr_dim_list, sr

def calc_sampen(files: np.ndarray, partitions: int = 1, orig: bool = False, mili_s: float = 0.1, mode: str = 'median') -> tuple[np.ndarray[np.float32], float]:
	"""
	Calcula los exponentes de Lyapunov (con `nolds.lyap_e()`) para una lista de nombres de archivos que contienen los audios de ravdess

	# Parametros

	files : np.ndarray[str]
		Es la lista de pistas de audio del cual se quiere conseguir sus pistas de audio
	partitions : int
		Representa la partición de audio con la que se trabajará. Es útil cambiarla cuando se quiera probar las funciones, sin tener que utilizar toda la pista de audio .\
		Por defecto `partitions = 1`
	orig : bool
		Indica si el programa retorne los datos originales antes del resampleo. Por defecto `orig = False`
	mili_s : float
		Es el periodo con el que se realizará el resampleo. Por defecto `mili_s = 0.1`
	mode : str
		Es el modo de interpolación que se utilizará para el resampleo. Puede ser únicamente "mean" o "median". Por defecto `mode = 'median'`

	# Retorno

	Retorna una tupla, donde la primera posición corresponde un arreglo multidimensional que contiene la métrica con `nolds.sampen()` y la segunda contiene el sample rate respectivo
	"""
	sampen_list = np.empty(files.shape, dtype='float32')

	for i, file in enumerate(files):
		y, sr = get_audio_record_ravdess(file, orig, mili_s, mode)
		sampen_list[i] = nolds.sampen(y[:len(y)//partitions])
	
	return sampen_list, sr


if __name__ == '__main__':
	file = np.array([r'Ravdess\audio_speech_actors_01-24\Actor_01\03-01-01-01-01-01-01.wav', '03-01-02-01-02-02-13.wav', '03-01-07-02-01-02-19.wav'])

	print(f'\n{calc_hurst_rs(file, 10) = }')
	print(f'\n{calc_dfa(file, 10) = }')
	print(f'\n{(temp := calc_lyap_e(file, 10)) = }')
	print(f'\n{calc_lyap_r(file, 10) = }')
	print(f'\n{calc_corr_dim(file, 10) = }')
	print(f'\n{calc_sampen(file, 10) = }\n')
	
	print(f'Lyap_e[:, 0]{temp[0][:, 0]}')
	print(f'Lyap_e[:, 0]{temp[0][:, 1]}')
	print(f'Lyap_e[:, 0]{temp[0][:, 2]}')
	print(f'Lyap_e[:, 0]{temp[0][:, 3]}')