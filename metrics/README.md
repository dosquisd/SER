# Cálculo de las métricas

En esta carpeta es donde están recopiladas los archivos después de haber ejecutado los archivos en `/calc_measures`, pero por redundancia, entonces existe una nueva carpeta llamada `/data` donde están la información de los datos ya está normalizado y, de paso, agrupado.

Los archivos que tienen el sufijo `1` se refiere a las métricas que están calculadas sobre las pistas de audio resampleadas, en cambio, si no existe ningún sufijo, significa que las métricas están calculadas sobre los datos originales. Por limitaciones físicas de nuestros equipos de computo, las métricas de `corr_dim` y `lyap_r` no pueden ser calculadas con los datos originales, por lo que, toca valerse únicamente por sus resultados con los datos resampleados.

A la hora de estar escribiendo este archivo, se han presentado algunos errores que no han permitido calcular las métricas de `corr_dim` y `lyap_r`.
