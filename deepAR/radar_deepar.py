import os

# Ajuste de compatibilidad para evitar el error de descriptores de Numpy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tf_keras as keras
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.stats as stats
import streamlit as st

# --- PARCHE DE EMERGENCIA PARA TFP ---
# Esto evita el choque de dimensiones entre Keras y TFP
from tensorflow.python.ops import array_ops


def model_dist(t):
    # Forzamos conversión limpia a tensor de 32 bits
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    loc = t[..., :1]
    scale = 1e-3 + tf.math.softplus(t[..., 1:])
    return tfp.distributions.Normal(loc=loc, scale=scale)


# --- CONFIGURACIÓN DE RUTAS ABSOLUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- INTERFAZ ---
st.set_page_config(page_title="Radar DeepAR", layout="centered")
st.title("📊 Radar de Probabilidades")

SYMBOL = st.text_input("Símbolo:", value="TSLA").upper().strip()
niveles_raw = st.text_input("Niveles (separados por coma):", "406.00, 416.38")
ejecutar = st.button("Calcular Probabilidades")


# --- LÓGICA DE CÁLCULO ---
def calcular_probabilidad_temporal(dist, target_price, precio_actual, scaler):
    try:
        mu_scaled = float(tf.reduce_mean(dist.mean()).numpy())
        sigma_scaled = float(tf.reduce_mean(dist.stddev()).numpy())

        rango_precio = scaler.data_range_[0]
        sigma_real = sigma_scaled * rango_precio

        # Z-Score ajustado a volatilidad institucional
        z_score = (target_price - precio_actual) / (sigma_real * 2.5 + 1e-6)

        if target_price > precio_actual:
            prob = (1 - stats.norm.cdf(z_score)) * 100
        else:
            prob = stats.norm.cdf(z_score) * 100

        eta_velas = int(abs(target_price - precio_actual) / (sigma_real + 1e-9))
        return round(float(prob), 1), eta_velas
    except:
        return 0.0, 0


# --- PROCESO PRINCIPAL ---
if ejecutar:
    MODEL_PATH = os.path.join(BASE_DIR, "models", SYMBOL, "deepAR_model.keras")
    SCALER_PATH = os.path.join(BASE_DIR, "models", SYMBOL, "scalerAR.gz")

    if not os.path.exists(MODEL_PATH):
        st.error(f"Archivo no encontrado: {MODEL_PATH}")
    else:
        try:
            with st.spinner("Cargando inteligencia de mercado..."):
                # Carga blindada
                custom_objects = {
                    "DistributionLambda": tfp.layers.DistributionLambda,
                    "model_dist": model_dist
                }

                model = keras.models.load_model(
                    MODEL_PATH,
                    custom_objects=custom_objects,
                    compile=False,
                    safe_mode=False
                )
                scaler = joblib.load(SCALER_PATH)

                # Mercado real
                df = yf.download(SYMBOL, period="5d", interval="5m", progress=False).dropna()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                actual_p = df['Close'].iloc[-1]
                data_scaled = scaler.transform(df[['Close', 'Volume']].values)

                # Inferencia
                input_data = np.expand_dims(data_scaled[-100:], axis=0).astype(np.float32)
                dist_pred = model(input_data)

                # Mostrar resultados
                st.markdown(f"### 🎯 Resultados para {SYMBOL}")
                st.write(f"Precio Actual: **${actual_p:.2f}**")

                if niveles_raw:
                    niveles = [float(n.strip()) for n in niveles_raw.split(",")]
                    res_list = []
                    for n in niveles:
                        p, e = calcular_probabilidad_temporal(dist_pred, n, actual_p, scaler)
                        res_list.append({
                            "Nivel": f"${n:.2f}",
                            "Probabilidad": f"{p}%",
                            "ETA Est.": f"{e} velas"
                        })
                    st.table(pd.DataFrame(res_list))

        except Exception as e:
            st.error(f"Error al procesar el modelo: {str(e)}")