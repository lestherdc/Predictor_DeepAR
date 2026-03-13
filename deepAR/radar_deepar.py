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


# --- PARCHE DE FUERZA BRUTA PARA DIMENSIONES ---
def model_dist(t):
    # Convertimos a tensor y extraemos los valores CRUDOS
    # Esto elimina cualquier metadato de Numpy que cause el error 'getset_descriptor'
    t = tf.convert_to_tensor(t, dtype=tf.float32)

    # Extraemos media y desviación estándar usando slicing de Tensores puro
    # Usamos tf.shape(t)[-1] para evitar que TFP use numpy.shape
    loc = t[..., :1]
    scale = 1e-3 + tf.math.softplus(t[..., 1:])

    return tfp.distributions.Normal(loc=loc, scale=scale)


# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Radar DeepAR", layout="centered")
st.title("📊 Radar de Probabilidades")

SYMBOL = st.text_input("Símbolo:", value="TSLA").upper().strip()
niveles_raw = st.text_input("Niveles (separados por coma):", "406.00, 416.38")
ejecutar = st.button("Calcular Probabilidades")


def calcular_probabilidad_temporal(dist, target_price, precio_actual, scaler):
    # Forzamos a que el cálculo sea con valores escalares de Python
    mu_scaled = float(np.mean(dist.mean()))
    sigma_scaled = float(np.mean(dist.stddev()))

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
            with st.spinner("Procesando DeepAR..."):
                # Cargamos el modelo inyectando la función parchada
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

                df = yf.download(SYMBOL, period="5d", interval="5m", progress=False).dropna()
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

                actual_p = df['Close'].iloc[-1]
                data_scaled = scaler.transform(df[['Close', 'Volume']].values)

                # OBLIGATORIO: Convertir a float32 y remover cualquier rastro de objeto numpy
                input_data = np.array(data_scaled[-100:], dtype=np.float32)
                input_data = np.expand_dims(input_data, axis=0)

                dist_pred = model(input_data)

                st.subheader(f"🎯 Análisis para {SYMBOL}")
                st.write(f"Precio Actual: **${actual_p:.2f}**")

                if niveles_raw:
                    niveles = [float(n.strip()) for n in niveles_raw.split(",")]
                    resultados = []
                    for n in niveles:
                        p, e = calcular_probabilidad_temporal(dist_pred, n, actual_p, scaler)
                        dist_pct = ((n - actual_p) / actual_p) * 100
                        resultados.append({
                            "Nivel": f"${n:.2f}",
                            "Distancia": f"{dist_pct:+.2f}%",
                            "Probabilidad": f"{p}%",
                            "ETA (velas)": e
                        })
                    st.table(pd.DataFrame(resultados))

        except Exception as e:
            st.error(f"Error persistente: {str(e)}")
            st.warning("Si el error persiste, intenta reiniciar el servidor de Streamlit desde el panel 'Manage App'.")