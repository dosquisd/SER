import os
import librosa
import numpy as np
from typing import TypeAlias

Emotion: TypeAlias = list[float] # Tiene las métricas respectiva para una emoción en concreto
Emotions: TypeAlias = list[Emotion] # Lista de emociones de una métrica en concreto

def get_feature_ravdess(file: str, feature: int) -> int | None:
	"""
	# Explicación

	Dentro del nombre de las pistas de audio en las grabaciones de Ravdess, hay diferentes caracteristicas. Por ejemplo,
	si se desea colocar `file = 02-01-06-01-02-01-12.wav`,es posible sacar lo siguiente:

	- Video-only (02)
	- Speech (01)
	- Fearful (06)
	- Normal intensity (01)
	- Statement "dogs" (02)
	- 1st Repetition (01)
	- 12th Actor (12) - Female (as the actor ID number is even)

	Dependiendo del `feature` que se quiera, la función arrojará `02` si se quiere saber si es video solo (para ese caso, feature = 1).

	# Parametros

	@param file: Es una cadena que contiene la pista de audio de la carpeta de Ravdess
	@param feature: Es un entero que especifica la feature que se necesita. Solo puede variar entre valores del 1 al 7, donde cada valor implica lo siguiente
	1. Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
	2. Vocal channel (01 = speech, 02 = song).
	3. Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
	4. Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
	5. Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
	6. Repetition (01 = 1st repetition, 02 = 2nd repetition).
	7. Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

	# Retorno

	La función finalmente arroja el número entero de la `feature` que se le especifique. Se retorna `None` cuando el feature no está en los limites establecidos
	o cuando no es un entero
	"""
	if (feature < 1 or feature > 7) or type(feature) != int:
		return None

	file = os.path.basename(file) # Es realizado únicamente por asegurar
	filename, _ = os.path.splitext(file)

	features = filename.split('-')
	return int(features[feature-1])

def get_mean_emotions(emotions) -> np.ndarray:
	"""
	# Explicación

	Obtiene la media para cada una de las emociones teniendo en cuenta la intensidad.
	
	# Parametros

	@param emotions. Es un arreglo que tiene 8 posiciones representando en cada una una emoción diferente, dentro de cada posición hay un arreglo
					 que contiene los datos para la intensidad normal y fuerte respectivamente.

	# Retorno

	Retorna un arreglo bidimensional de NumPy con el promedio de cada emoción, dentro de cada emoción hay un arreglo que diferencia el promedio
	de las emociones con intensidades normales con las fuertes
	"""

	averages = np.zeros(8, dtype=np.ndarray)
	for i, emotion in enumerate(emotions):
		averages[i] = np.mean(emotion, axis=1, dtype=np.float32)
	
	return averages

def get_std_emotions(emotions) -> np.ndarray:
	"""
	# Explicación

	Obtiene la media para cada una de las emociones teniendo en cuenta la intensidad. Funciona de manera análoga a `get_mean_emotions`, solo que 
	aquí se calcula la desviación estandar.
	
	# Parametros

	@param emotions. Es un arreglo que tiene 8 posiciones representando en cada una una emoción diferente, dentro de cada posición hay un arreglo
	que contiene los datos para la intensidad normal y fuerte respectivamente.

	# Retorno

	Retorna un arreglo bidimensional de NumPy con la desviación estandar de cada emoción, dentro de cada emoción hay un arreglo que diferencia el promedio
	de las emociones con intensidades normales con las fuertes
	"""

	stds: np.ndarray = np.zeros(8, dtype=np.ndarray)
	for i, emotion in enumerate(emotions):
		stds[i] = np.std(emotion, axis=1, dtype=np.float32)
	
	return stds

def alternate_lists(list1, list2) -> np.ndarray:
	"""Sea una lista1 = [1, 2, 3] y lista2 = [4, 5, 6], con la función se unen las listas alternandose entre ellas, es decir
	resulta en una lista como la siguiente: [1, 4, 2, 5, 3, 6]. Además de eso, toma las primeras dos posiciones y las convierte en una tupla
	para dejar el siguiente resultado: alternate_lists(list1, list2) = [(1, 4), (2, 5), (3, 6)]. lista1 y lista2 deben tener el mismo tamaño"""

	size = len(list1)
	return np.array([(list1[i], list2[i]) for i in range(size)])

def get_crests_valleys_ravdess(actor: str) -> tuple[Emotions]:
	"""
	# Explicación

	Cada uno de los actores tiene grabaciones para todas las emociones, esta función lo que hace es ver la onda del sonido y guardar la amplitud máxima, mínima
	(crestas y valles respectivamente) y la media en diferentes listas, es posible porque la onda no es completamente simétrica. Las amplitudes están guardadas teniendo 
	en cuenta la emoción respectiva. SOLO PARA EL DATASET DE RAVDESS
	
	Por ejemplo, para el actor01, las amplitudes máximas para los audios que tienen emoción neutra es la siguiente:

	`crests[0] = array([[0.04100873, 0.04718515, 0.05874421, 0.06183686],
       					[0.        , 0.        , 0.        , 0.        ]])`
	
	La primera fila corresponde a la intensidad `normal` y la segunda a la `strong`, como neutral solo tiene intensidad normal, entonces strong está lleno de 0
	
				   
	Para una emoción calmada, es la siguiente:

	`crests[1] = array([[0.02603629, 0.02865214, 0.0382348 , 0.04739437],
       					[0.02244095, 0.02698043, 0.02205249, 0.02993276]])`

	Y así para cada una de las emociones. Como se mencionó, hay una lista para las crestas, otra diferente para los valles y para el promedio.

	El orden en que están guardadas las emociones en todas las listas, es el mismo de cómo fueron mostrados en el DataSet de Ravdess, es decir,
	1. Neutral - 2. Calm - 3. Happy - 4. Sad - 5. Angry - 6. Fearful - 7. Disgust - 8. Surprised

	# Parametros

	@param actor: str. Es una cadena que contiene el nombre del actor del que se está tratando. Por ejemplo, para el primer actor, `actor` debería tener el siguiente
	valor: `actor = 'Actor_01'`, y así para cada uno de los actores

	# Retorno

	La función al final retorna las dos listas que contiene las crestas, los valles y la media de los audios en el mismo orden, Para cada una de estas 3 métricas, tienen
	dos arreglos de los cuales diferencian los datos con intensidad normal y fuerte.
	"""
	
	ravdess_path: str = r'Ravdess\audio_speech_actors_01-24'
	actor_path: str = fr'{ravdess_path}\{actor}'
	
	crests_normal, crests_strong = [[] for _ in range(8)], [[] for _ in range(8)]
	valleys_normal, valleys_strong = [[] for _ in range(8)], [[] for _ in range(8)]
	averages_normal, averages_strong = [[] for _ in range(8)], [[] for _ in range(8)]

	for record in os.listdir(actor_path):
		record_path: str = fr'{actor_path}\{record}'
		y: np.ndarray = librosa.load(record_path)[0]

		index: int = get_feature_ravdess(record, 3) - 1

		if index == 0: # Si la emoción es neutra, entonces no hay diferentes intensidades
			crests_normal[0].append(np.max(y))
			valleys_normal[0].append(np.min(y))
			averages_normal[0].append(np.mean(y))

			crests_strong[0].append(0)
			valleys_strong[0].append(0)
			averages_strong[0].append(0)

			continue

		if get_feature_ravdess(record, 4) == 1: # Si la intensidad es `normal`
			crests_normal[index].append(np.max(y))
			valleys_normal[index].append(np.min(y))
			averages_normal[index].append(np.mean(y))

		else: # Si la intensidad es `strong`
			crests_strong[index].append(np.max(y))
			valleys_strong[index].append(np.min(y))
			averages_strong[index].append(np.mean(y))

	# crests y demás están organizando de esta manera: [[crests_normal[0], crests_strong[0]], ...]	
	crests = alternate_lists(crests_normal, crests_strong)
	valleys = alternate_lists(valleys_normal, valleys_strong)
	averages = alternate_lists(averages_normal, averages_strong)

	return crests, valleys, averages

if __name__ == '__main__':
	a = get_crests_valleys_ravdess('Actor_01')[0]
	print(a, '\n\n')
	print(get_mean_emotions(a), '\n\n')
	print(get_std_emotions(a))