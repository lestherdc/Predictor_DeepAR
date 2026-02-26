import os
import yfinance as yf
import pandas as pd
import joblib
from src.data_engine import DeepARDDataEngine
from src.model_deepar import build_deepar_model

# ------ Nuestra configuracion --------
ACCIONES = ['PLTR','AAPL','TSLA']
PERIOD = "60d"
INTERVAL = "5m"
WINDOWS_SIZE = 60
EPOCHS = 15
BATCH_SIZE = 32

def start_training():
    engine = DeepARDDataEngine(window_size=WINDOWS_SIZE)

    for symbol in ACCIONES:
        print(f"\n" + "="*50)
        print(f"🏗️  ENTRENANDO MODELO DEEPAR PARA: {symbol}")
        print("=" * 50)

    #1. creacion de las carpetas de salida
    model_path = f"models/{symbol}/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    #2. Descarga de datos
    print(f"📥 Descargando {symbol}...")
    df = yf.download(symbol, period=PERIOD, interval=INTERVAL, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    #3. Procesamiento de Probabilidades
    #DeepAR necesito la secuencia y el valor siguiente
    X, y = engine.prepare_data(df, training=True)

    #Guardamos el scaler para uso futuro en el Radar
    scaler_filename = f"{model_path}/scaler.gz"
    joblib.dump(engine.get_scaler(), scaler_filename)
    print(f"✅ Scaler guardado en {scaler_filename}")

    #4, Construccion del Modelo
    # X.shape[2] son las caracteristicas (Close y volumen)
    model = build_deepar_model(WINDOWS_SIZE, X.shape[2])

    #5. Entrenamiento
    print(f"🧠 Ajustando curva de probabilidad...")
    model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_split=0.1,
    )

    #6. Guardar pesos
    model_save_path = f"{model_path}/deepar_model.keras"
    model.save(model_save_path)
    print(f"💾 Modelo {symbol} guardado exitosamente!")

if __name__ == "__main__":
    start_training()