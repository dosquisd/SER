import pandas as pd
from my_functions.calc_nolds_metrics import *
from copy import deepcopy
from time import time


def calc(df: pd.DataFrame, func: callable, label: str, filename: str, partitions: int = 1, emb_dim: int = 14, lyap_e: bool = False) -> None:
	files = df['File'].to_numpy()

	if not lyap_e:
		try:
			df[label] = func(files, partitions)
		except Exception as e:
			print(f'Error: {repr(e)}')
		finally:
			df.to_csv(filename, sep=';')
			return None
		
	try:
		lyap_e = calc_lyap_e(files, partitions)
		df['1st Lyapunov Exponent'] = lyap_e[:, 0]
		df['2nd Lyapunov Exponent'] = lyap_e[:, 1]
		df['3rd Lyapunov Exponent'] = lyap_e[:, 2]
		df['4th Lyapunov Exponent'] = lyap_e[:, 3]
	except Exception as e:
		print(f'Error: {repr(e)}')
	finally:
		df.to_csv(filename, sep=';')
		return None


records_df = pd.read_csv('data.csv', delimiter=';', index_col=0)
partitions = 50

print('Sample Entropy')
t0 = time()
calc(deepcopy(records_df), calc_sampen, 'Sample Entropy', 'sampen.csv', partitions)
print(f'Tiempo: {time() - t0} s')

print('\nCorrelation Dimension')
t0 = time()
calc(deepcopy(records_df), calc_corr_dim, 'Correlation Dimension', 'corr_dim.csv', partitions)
print(f'Tiempo: {time() - t0} s')

print('\nHurst Exponent')
t0 = time()
calc(deepcopy(records_df), calc_hurst_rs, 'Hurst Exponent', 'hurst.csv', partitions)
print(f'Tiempo: {time() - t0} s')

print('\nDfa')
t0 = time()
calc(deepcopy(records_df), calc_dfa, 'Detrented Fluctuation Analysis', 'dfa.csv', partitions)
print(f'Tiempo: {time() - t0} s')

print('\nLyapunov Exponents')
t0 = time()
calc(deepcopy(records_df), calc_lyap_e, '', 'lyap_e.csv', partitions, True)
print(f'Tiempo: {time() - t0} s')
