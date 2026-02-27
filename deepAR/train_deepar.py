import os
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import tf_keras as keras
from brain_deepar import DeepARModel

# --- CONFIGURACIÓN ---
SYMBOL = "PLTR"
WINDOW_SIZE = 100
EPOCHS = 50
BATCH_SIZE = 32
MODEL_DIR = f"models/{SYMBOL}"
os.makedirs(MODEL_DIR, exist_ok=True)


def preparar_datos(symbol):
    print(f"📥 Descargando datos históricos para {symbol}...")
    df = yf.download(symbol, period="60d", interval="5m").dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    data = df[['Close', 'Volume']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(WINDOW_SIZE, len(data_scaled)):
        X.append(data_scaled[i - WINDOW_SIZE:i])
        y.append(data_scaled[i, 0])

    return np.array(X), np.array(y), scaler


def entrenar():
    X, y, scaler = preparar_datos(SYMBOL)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    brain = DeepARModel(window_size=WINDOW_SIZE, feature_count=2)
    model = brain.build_model()

    print(f"🚀 Iniciando entrenamiento (Epochs: {EPOCHS})...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # GUARDADO CON NOMBRES EXCLUSIVOS
    model.save(f"{MODEL_DIR}/deepAR_model.keras")
    joblib.dump(scaler, f"{MODEL_DIR}/scalerAR.gz")
    print(f"\n✅ DeepAR entrenado y guardado en {MODEL_DIR}")


if __name__ == "__main__":
    entrenar()