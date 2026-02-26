import os

# --- CONFIGURACIÓN DE COMPATIBILIDAD ---
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_KERAS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tf_keras as keras
import tensorflow as tf
from src.data_engine import DeepARDDataEngine
from src.model_deepar import build_deepar_model

# --- CONFIGURACIÓN DE ENTRENAMIENTO ---
ACCIONES = ["PLTR", "AAPL", "TSLA", "NVDA"]
PERIOD = "60d"
INTERVAL = "5m"
WINDOW_SIZE = 100
EPOCHS = 40
BATCH_SIZE = 32


def start_training():
    engine = DeepARDDataEngine(window_size=WINDOW_SIZE)

    for symbol in ACCIONES:
        try:
            print(f"\n" + "=" * 50)
            print(f"🏗️  INICIANDO PROCESO PARA: {symbol}")
            print("=" * 50)

            # 1. Crear carpetas de salida si no existen
            model_path = f"models/{symbol}"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
                print(f"📁 Carpeta creada: {model_path}")

            # 2. Descarga de datos con reintentos
            print(f"📥 Descargando datos de {symbol}...")
            df = yf.download(symbol, period=PERIOD, interval=INTERVAL, progress=False)

            if df.empty:
                print(f"⚠️ No se obtuvieron datos para {symbol}. Saltando...")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()

            # 3. Procesamiento
            X, y = engine.prepare_data(df, training=True)

            # Guardar el scaler
            scaler_filename = f"{model_path}/scaler.gz"
            joblib.dump(engine.get_scaler(), scaler_filename)
            print(f"✅ Scaler guardado.")

            # 4. Construcción y Entrenamiento
            print(f"🧠 Entrenando cerebro probabilístico (Epochs: {EPOCHS})...")

            # Limpiar sesión anterior para evitar saturación de memoria
            keras.backend.clear_session()

            model = build_deepar_model(WINDOW_SIZE, X.shape[2])

            model.fit(
                X, y,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=1,
                validation_split=0.1
            )

            # 5. Guardado
            model_save_path = f"{model_path}/deepar_model.keras"
            model.save(model_save_path)
            print(f"💾 ¡Modelo {symbol} guardado exitosamente!")

        except Exception as e:
            print(f"❌ ERROR CRÍTICO EN {symbol}: {str(e)}")
            print("⏭️ Saltando a la siguiente acción en la lista...")
            continue

    print(f"\n✅ PROCESO FINALIZADO. Revisa tu carpeta 'models/'")


if __name__ == "__main__":
    start_training()