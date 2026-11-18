# Asistencia de estudiantes por reconocimiento de voz

Práctica del segundo parcial de Machine Learning.

- **Área:** Reconocimiento de voz
- **Subárea:** Reconocimiento de hablante (*speaker recognition*)
- **Problema:** registrar la asistencia identificando al estudiante por su voz

Se comparan dos modelos sobre el mismo problema:

| Modelo | Tipo | Entrada |
|---|---|---|
| CNN (y LSTM opcional) | Aprendizaje profundo | Espectrograma log-Mel (128 × 63) |
| K-Means + jerárquico + DBSCAN | No supervisado | Vector MFCC (media + desviación, 80 valores) |

## Estructura

```
data/raw/<estudiante>/*.wav   audios originales (no se suben a GitHub)
data/processed/dataset.npz    dataset preprocesado
models/                       modelos entrenados
reports/                      gráficas y métricas para el informe
src/config.py                 parámetros (frecuencia, duración, umbrales)
src/audio_utils.py            limpieza, segmentación, aumento y características
src/predictor.py              identificación con cualquiera de los modelos
scripts/01_grabar_dataset.py  grabar audios desde el micrófono
scripts/02_preprocesar.py     V.a  preprocesamiento del dataset
scripts/03_entrenar_cnn.py    V.b/V.c  modelo de deep learning
scripts/04_clustering.py      V.b/V.c  modelo no supervisado
scripts/05_comparar.py        tabla comparativa de los modelos
app/app.py                    prototipo en Streamlit
```

## Instalación (una vez por computadora)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si PowerShell bloquea la activación: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Uso

Con el entorno activado y desde la raíz del proyecto:

```powershell
# 1. Grabar (repetir por cada estudiante; la sesión 2 en otro día)
python scripts/01_grabar_dataset.py --nombre juan_perez --sesion 1 --cantidad 30

# 2. Preprocesar
python scripts/02_preprocesar.py

# 3. Entrenar el modelo profundo (y opcionalmente la LSTM)
python scripts/03_entrenar_cnn.py
python scripts/03_entrenar_cnn.py --modelo lstm

# 4. Entrenar el modelo no supervisado
python scripts/04_clustering.py

# 5. Comparar
python scripts/05_comparar.py

# 6. Abrir la aplicación
streamlit run app/app.py
```

También se pueden agregar audios de un dataset público (por ejemplo Mozilla Common
Voice en español): basta con copiar los archivos de cada hablante en
`data/raw/<id_hablante>/`.

## Recomendaciones para el dataset

- 15 a 30 personas, 20 a 40 audios de unos 3 s cada una.
- Dos sesiones en días distintos (`--sesion 1` y `--sesion 2`): si todos tienen
  sesión 2, se usa automáticamente como conjunto de prueba.
- Pedir consentimiento a quienes graban. Los audios **no** se suben al repositorio.
