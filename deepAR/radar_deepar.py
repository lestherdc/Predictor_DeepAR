import os

# Silenciamos advertencias innecesarias
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


# --- PARCHE TÉCNICO ---
def model_dist(t):
    # Forzamos la conversión a tensor puro de TensorFlow
    # Esto "desconecta" el tensor de cualquier lógica de Numpy que cause el error
    inputs = tf.cast(t, tf.float32)
    loc = inputs[..., :1]
    scale = 1e-3 + tf.math.softplus(inputs[..., 1:])
    return tfp.distributions.Normal(loc=loc, scale=scale)


# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- INTERFAZ STREAMLIT ---
st.set_page_config(page_title="Radar DeepAR", layout="centered")
st.title("📊 Radar de Probabilidades")

SYMBOL = st.text_input("Símbolo:", value="TSLA").upper().strip()
niveles_raw = st.text_input("Niveles (separados por coma):", "406.00, 416.38")
ejecutar = st.button("Calcular Probabilidades")


def calcular_probabilidad_temporal(dist, target_price, precio_actual, scaler):
    # Usamos tf.gather para extraer valores sin tocar Numpy directamente
    mu_scaled = tf.reduce_mean(dist.mean()).numpy()
    sigma_scaled = tf.reduce_mean(dist.stddev()).numpy()

    rango_precio = scaler.data_range_[0]
    mu_real = (mu_scaled * rango_precio) + scaler.data_min_[0]
    sigma_real = sigma_scaled * rango_precio

    # Z-Score (Volatilidad 2.5 sigmas)
    z_score = (target_price - precio_actual) / (sigma_real * 2.5 + 1e-6)

    if target_price > precio_actual:
        prob = (1 - stats.norm.cdf(z_score)) * 100
    else:
        prob = stats.norm.cdf(z_score) * 100

    eta = int(abs(target_price - precio_actual) / (sigma_real + 1e-9))
    return round(float(prob), 1), eta


if ejecutar:
    MODEL_PATH = os.path.join(BASE_DIR, "models", SYMBOL, "deepAR_model.keras")
    SCALER_PATH = os.path.join(BASE_DIR, "models", SYMBOL, "scalerAR.gz")

    if not os.path.exists(MODEL_PATH):
        st.error(f"No encontré el modelo en: {MODEL_PATH}")
    else:
        try:
            with st.spinner(f"Cargando modelo de {SYMBOL}..."):
                # Cargamos el modelo con los custom_objects corregidos
                model = keras.models.load_model(
                    MODEL_PATH,
                    custom_objects={
                        "DistributionLambda": tfp.layers.DistributionLambda,
                        "model_dist": model_dist
                    },
                    compile=False,
                    safe_mode=False
                )
                scaler = joblib.load(SCALER_PATH)

                # Datos de Yahoo Finance
                df = yf.download(SYMBOL, period="5d", interval="5m", progress=False).dropna()
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

                actual_p = df['Close'].iloc[-1]

                # Inferencia
                data_scaled = scaler.transform(df[['Close', 'Volume']].values)
                # Forzamos float32 aquí también para evitar el descriptor de numpy
                input_data = np.expand_dims(data_scaled[-100:], axis=0).astype(np.float32)

                dist_pred = model(input_data)

                # Resultados
                st.subheader(f"🎯 Análisis para {SYMBOL}")
                st.write(f"Precio Actual: **${actual_p:.2f}**")

                if niveles_raw:
                    niveles = [float(n.strip()) for n in niveles_raw.split(",")]
                    resultados = []
                    for n in niveles:
                        p, e = calcular_probabilidad_temporal(dist_pred, n, actual_p, scaler)
                        resultados.append({
                            "Nivel": f"${n:.2f}",
                            "Probabilidad": f"{p}%",
                            "ETA (velas)": e
                        })
                    st.table(pd.DataFrame(resultados))

        except Exception as e:
            st.error(f"Error técnico: {str(e)}")
            st.info("Sugerencia: Verifica que tu requirements.txt tenga numpy==1.26.4")