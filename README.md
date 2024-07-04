# Speech Emotion Recognition (SER)

Este es un proyecto que se está trabajando de la mano con el profesor David Sierra Porta, en el que se busca hacer un clasificador de las emociones en base en el cálculo de medidas topológicas usando el módulo [`nolds`](https://github.com/CSchoel/nolds).

Hasta el momento, la limitante más grande que se ha tenido es con nuestros equipos de computo, pues para el caso del cálculo de las métricas `corr_dim` y `lyap_r` se requiere más RAM de la que se tiene, pero en pro de no estar varados, se decidió realizar un resampleo en las pistas de audio cada `0.1 ms` utilizando la mediana (ver [resamples.ipynb](resamples.ipynb)), aún así, para comparar qué tan distantes están los resultados de las métricas con las pistas de audio resampleadas y no está [`resamplesComparisons.ipynb`](resamplesComparisons.ipynb).
