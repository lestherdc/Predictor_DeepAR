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


# --- 1. FUNCIÓN DE DISTRIBUCIÓN PURA (SIN DEPENDENCIAS DE NUMPY) ---
def model_dist(t):
    # Forzamos la extracción manual de valores para evitar que TFP busque el atributo .shape
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    # t[..., :1] es loc, t[..., 1:] es scale
    loc = tf.gather(t, [0], axis=-1)
    scale = 1e-3 + tf.math.softplus(tf.gather(t, [1], axis=-1))
    return tfp.distributions.Normal(loc=loc, scale=scale)


# --- 2. CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Radar DeepAR", layout="centered")
st.title("📊 Radar de Probabilidades")

SYMBOL = st.text_input("Símbolo:", value="TSLA").upper().strip()
niveles_raw = st.text_input("Niveles (separados por coma):", "406.00, 416.38")
ejecutar = st.button("Calcular Probabilidades")


def calcular_probabilidad_temporal(dist, target_price, precio_actual, scaler):
    # Extraemos media y desviación usando métodos puramente de TensorFlow
    mu_scaled = tf.reduce_mean(dist.mean()).numpy()
    sigma_scaled = tf.reduce_mean(dist.stddev()).numpy()

    rango_precio = scaler.data_range_[0]
    mu_real = (mu_scaled * rango_precio) + scaler.data_min_[0]
    sigma_real = sigma_scaled * rango_precio

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
        st.error(f"Modelo no encontrado en: {MODEL_PATH}")
    else:
        try:
            # --- 3. CARGA EXPERIMENTAL (IGNORANDO CONFIGURACIÓN DE CAPA) ---
            custom_objects = {
                "DistributionLambda": tfp.layers.DistributionLambda,
                "model_dist": model_dist
            }

            # Cargamos el modelo
            # Si el error persiste, Keras está leyendo la función lambda original del archivo.
            # safe_mode=False es vital aquí.
            model = keras.models.load_model(
                MODEL_PATH,
                custom_objects=custom_objects,
                compile=False,
                safe_mode=False
            )
            scaler = joblib.load(SCALER_PATH)

            # --- 4. OBTENCIÓN DE DATOS Y PREDICCIÓN ---
            df = yf.download(SYMBOL, period="5d", interval="5m", progress=False).dropna()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            actual_p = df['Close'].iloc[-1]
            data_scaled = scaler.transform(df[['Close', 'Volume']].values)

            # El secreto: Convertir a tensor de TF ANTES de pasar al modelo
            input_window = data_scaled[-100:].astype(np.float32)
            input_tensor = tf.convert_to_tensor(np.expand_dims(input_window, axis=0))

            # Realizamos la inferencia pasándole un Tensor, no un array de Numpy
            dist_pred = model(input_tensor)

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
            st.error(f"Error persistente en TFP: {str(e)}")
            st.info("Este error ocurre por una incompatibilidad de Numpy en el servidor.")