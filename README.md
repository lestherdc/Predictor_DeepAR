# Predictor DeepAR
Este predictor fue desarrolado por Amazon, el cual en simples palabras
predice el rango en el cual estara el precio de una accion

## Estructura del proyecto
DeepAR_Forecast/
├── data/               # Archivos CSV descargados (opcional)
├── models/             # Pesos del modelo (.keras) y Scalers (.bin)
│   ├── TSLA/
│   └── PLTR/
├── src/                # El motor del sistema
│   ├── __init__.py
│   ├── data_engine.py  # Procesamiento de series temporales
│   ├── model_deepar.py # Arquitectura probabilística
│   └── radar_logic.py  # Tu lógica de niveles históricos
├── trainer.py          # Script para entrenar múltiples acciones
└── main_radar.py       # Ejecución en vivo y visualización