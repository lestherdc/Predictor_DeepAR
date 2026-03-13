import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tf_keras as keras
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.stats as stats
import streamlit as st


# --- 1. ARQUITECTURA DETERMINISTA (EVITA EL ERROR DE TFP) ---
def build_radar_model(input_shape=(100, 2)):
    """
    Construimos el modelo sin la capa DistributionLambda para evitar el error de dimensiones.
    """
    inputs = keras.Input(shape=input_shape)
    # Ajusta las unidades (64) a las que usaste en tu entrenamiento
    x = keras.layers.LSTM(64, return_sequences=False)(inputs)
    # La última capa Dense tiene 2 unidades: una para la media y otra para la desviación
    outputs = keras.layers.Dense(2)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


# --- 2. CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Radar DeepAR", layout="centered")
st.title("📊 Radar de Probabilidades (Solución Final)")

SYMBOL = st.text_input("Símbolo:", value="TSLA").upper().strip()
niveles_raw = st.text_input("Niveles:", "406.00, 416.38")
ejecutar = st.button("Calcular Probabilidades")


# --- 3. LÓGICA DE INFERENCIA MANUAL ---
def calcular_probabilidades_manual(model_output, target_price, precio_actual, scaler):
    # Separamos manualmente la salida del modelo (Media y Desviación)
    # model_output tiene forma [1, 2]
    mu_scaled = model_output[0, 0]
    # Aplicamos softplus manualmente como lo hacía el modelo original
    sigma_raw = model_output[0, 1]
    sigma_scaled = 1e-3 + tf.math.softplus(sigma_raw).numpy()

    rango_precio = scaler.data_range_[0]
    sigma_real = sigma_scaled * rango_precio

    # Cálculo de Z-Score (2.5 sigmas para volatilidad institucional)
    z_score = (target_price - precio_actual) / (sigma_real * 2.5 + 1e-6)

    if target_price > precio_actual:
        prob = (1 - stats.norm.cdf(z_score)) * 100
    else:
        prob = stats.norm.cdf(z_score) * 100

    eta = int(abs(target_price - precio_actual) / (sigma_real + 1e-9))
    return round(float(prob), 1), eta


# --- 4. EJECUCIÓN ---
if ejecutar:
    MODEL_PATH = os.path.join(BASE_DIR, "models", SYMBOL, "deepAR_model.keras")
    SCALER_PATH = os.path.join(BASE_DIR, "models", SYMBOL, "scalerAR.gz")

    if not os.path.exists(MODEL_PATH):
        st.error(f"No se encontró el modelo para {SYMBOL}")
    else:
        try:
            with st.spinner("Realizando inferencia probabilística..."):
                # Reconstruimos y cargamos pesos (ignora la capa problemática)
                model = build_radar_model()
                model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)

                scaler = joblib.load(SCALER_PATH)

                # Datos de mercado
                df = yf.download(SYMBOL, period="5d", interval="5m", progress=False).dropna()
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

                actual_p = df['Close'].iloc[-1]
                data_scaled = scaler.transform(df[['Close', 'Volume']].values)

                # Inferencia determinista
                input_data = np.expand_dims(data_scaled[-100:], axis=0).astype(np.float32)
                raw_pred = model.predict(input_data)  # Devuelve [media_scaled, sigma_scaled_raw]

                # Procesar niveles
                if niveles_raw:
                    niveles = [float(n.strip()) for n in niveles_raw.split(",")]
                    res = []
                    for n in niveles:
                        p, e = calcular_probabilidades_manual(raw_pred, n, actual_p, scaler)
                        res.append({
                            "Objetivo": f"${n:.2f}",
                            "Probabilidad": f"{p}%",
                            "ETA (velas)": e
                        })
                    st.table(pd.DataFrame(res))

        except Exception as e:
            st.error(f"Error en Radar: {str(e)}")