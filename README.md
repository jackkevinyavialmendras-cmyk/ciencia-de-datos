# Asistencia de estudiantes por reconocimiento de voz

Práctica del segundo parcial de Machine Learning.

- **Área:** Reconocimiento de voz
- **Subárea:** Reconocimiento de hablante (*speaker recognition*)
- **Problema:** registrar la asistencia identificando al estudiante por su voz
- **Enfoque:** aprendizaje **no supervisado**

Cada audio se convierte en un vector MFCC (media + desviación de 40 coeficientes = 80
valores). Los algoritmos agrupan las voces por parecido **sin usar etiquetas**:

| Algoritmo | Tipo | Uso |
|---|---|---|
| **K-Means** | Agrupamiento por particiones | Modelo principal; k con método del codo y silhouette |
| **Jerárquico aglomerativo** | Clustering jerárquico | Dendrogramas comparando MIN, MAX, Promedio y Centroides |
| **DBSCAN** | Basado en densidad | Detección de ruido / outliers; eps con gráfico k-distancia |

Las etiquetas reales solo se usan al final para evaluar (pureza, ARI, NMI) y para nombrar
cada cluster con el estudiante mayoritario.

## Estructura

```
data/raw/<estudiante>/*.wav     audios originales (se suben para que el grupo pueda probar)
data/desconocidos/<persona>/    voces NO registradas, para probar el rechazo
data/processed/dataset.npz      dataset preprocesado (vectores MFCC)
models/clustering.joblib        modelo K-Means entrenado
reports/                        gráficas y métricas para el informe
src/config.py                   parámetros (frecuencia, duración, umbrales)
src/audio_utils.py              limpieza, segmentación, MFCC y control de calidad
src/predictor.py                identificación de una voz nueva con K-Means
scripts/01_grabar_dataset.py    grabar audios desde el micrófono (consola)
scripts/02_preprocesar.py       V.a  preprocesamiento del dataset
scripts/03_clustering.py        V.b/V.c  K-Means, jerárquico, DBSCAN y validación
app/                            interfaz web (Streamlit)
iniciar.bat                     abre la interfaz con doble clic
```

## Instalación (una vez por computadora)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si PowerShell bloquea la activación: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Uso con interfaz (recomendado)

Doble clic en **`iniciar.bat`** (o `streamlit run app/app.py`) y abrir
**http://localhost:8501** en Chrome. Desde el menú lateral se hace todo:

| Página | Qué hace |
|---|---|
| Inicio | Estado del proyecto paso a paso |
| Grabar voces | Grabación guiada frase por frase, revisa la calidad y guarda sola; también permite subir audios del celular y grabar voces desconocidas |
| Revisar dataset | Cuántos audios tiene cada persona, avisos (desequilibrio, falta sesión 2), escuchar y borrar |
| Agrupar voces | Ejecuta el preprocesamiento y el clustering con la salida en vivo |
| Resultados | Métricas, tabla comparativa, codo/silhouette, PCA, dendrogramas y k-distancia |
| Tomar asistencia | Identifica la voz y marca PRESENTE; incluye prueba con voces desconocidas |
| Reporte | Asistencia por fecha, descarga en Excel |

Chrome pedirá permiso para usar el micrófono: hay que aceptar. El micrófono solo
funciona en `localhost` (la misma computadora); desde otro dispositivo por la red el
navegador lo bloquea por no ser HTTPS, así que en ese caso usen «Subir audios».

## Uso por línea de comandos

Con el entorno activado y desde la raíz del proyecto:

```powershell
# 1. Grabar (repetir por cada estudiante; la sesión 2 en otro día)
python scripts/01_grabar_dataset.py --nombre juan_perez --sesion 1 --cantidad 25

# 2. Preprocesar
python scripts/02_preprocesar.py

# 3. Clustering: K-Means, jerárquico y DBSCAN
python scripts/03_clustering.py

# 4. Abrir la aplicación
streamlit run app/app.py
```

## Recomendaciones para el dataset

- 15 a 30 personas, 20 a 40 audios de unos 3 s cada una.
- Dos sesiones en días distintos (`--sesion 1` y `--sesion 2`): si todos tienen
  sesión 2, se usa automáticamente como conjunto de prueba.
- Pedir consentimiento a quienes graban: los audios se suben al repositorio para que todo el grupo pueda probar el sistema.
