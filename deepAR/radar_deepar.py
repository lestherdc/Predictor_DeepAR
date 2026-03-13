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


# --- 1. ARQUITECTURA MANUAL (BYPASS AL ARCHIVO .KERAS) ---
def model_dist(t):
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    loc = t[..., :1]
    scale = 1e-3 + tf.math.softplus(t[..., 1:])
    return tfp.distributions.Normal(loc=loc, scale=scale)


def build_deepar_model(input_shape=(100, 2)):
    """
    Reconstruye la arquitectura exacta de tu Radar DeepAR.
    Ajusta las unidades de LSTM (ej: 64) según como lo entrenaste.
    """
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.LSTM(64, return_sequences=False)(inputs)  # Cambia 64 por tus unidades originales
    x = keras.layers.Dense(2)(x)  # Salida para Media y Desviación
    outputs = tfp.layers.DistributionLambda(model_dist)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


# --- 2. CONFIGURACIÓN Y RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Radar DeepAR", layout="centered")
st.title("📊 Radar de Probabilidades (Modo Seguro)")

SYMBOL = st.text_input("Símbolo:", value="TSLA").upper().strip()
niveles_raw = st.text_input("Niveles:", "406.00, 416.38")
ejecutar = st.button("Calcular Probabilidades")


# --- 3. LÓGICA DE CÁLCULO ---
def calcular_probabilidad_temporal(dist, target_price, precio_actual, scaler):
    mu_scaled = float(tf.reduce_mean(dist.mean()).numpy())
    sigma_scaled = float(tf.reduce_mean(dist.stddev()).numpy())
    rango_precio = scaler.data_range_[0]
    sigma_real = sigma_scaled * rango_precio

    z_score = (target_price - precio_actual) / (sigma_real * 2.5 + 1e-6)
    prob = (1 - stats.norm.cdf(z_score)) * 100 if target_price > precio_actual else stats.norm.cdf(z_score) * 100
    eta = int(abs(target_price - precio_actual) / (sigma_real + 1e-9))
    return round(float(prob), 1), eta


# --- 4. EJECUCIÓN PRINCIPAL ---
if ejecutar:
    MODEL_PATH = os.path.join(BASE_DIR, "models", SYMBOL, "deepAR_model.keras")
    SCALER_PATH = os.path.join(BASE_DIR, "models", SYMBOL, "scalerAR.gz")

    if not os.path.exists(MODEL_PATH):
        st.error("No se encontró el archivo del modelo.")
    else:
        try:
            with st.spinner("Reconstruyendo modelo y cargando pesos..."):
                # PASO CLAVE: Reconstruimos la red y solo inyectamos los pesos del archivo
                model = build_deepar_model()

                # Cargamos los pesos del archivo .keras (esto ignora la configuración de la capa DistributionLambda rota)
                model.load_weights(MODEL_PATH)

                scaler = joblib.load(SCALER_PATH)

                # Datos de mercado
                df = yf.download(SYMBOL, period="5d", interval="5m", progress=False).dropna()
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

                actual_p = df['Close'].iloc[-1]
                data_scaled = scaler.transform(df[['Close', 'Volume']].values)

                # Inferencia con Tensor limpio
                input_data = tf.convert_to_tensor(np.expand_dims(data_scaled[-100:], axis=0), dtype=tf.float32)
                dist_pred = model(input_data)

                st.subheader(f"🎯 Resultados para {SYMBOL}")
                st.write(f"Precio Actual: **${actual_p:.2f}**")

                if niveles_raw:
                    niveles = [float(n.strip()) for n in niveles_raw.split(",")]
                    res = [{"Nivel": f"${n:.2f}",
                            "Probabilidad": f"{calcular_probabilidad_temporal(dist_pred, n, actual_p, scaler)[0]}%",
                            "ETA": calcular_probabilidad_temporal(dist_pred, n, actual_p, scaler)[1]} for n in niveles]
                    st.table(pd.DataFrame(res))

        except Exception as e:
            st.error(f"Error en reconstrucción: {str(e)}")
            st.info("Asegúrate de que las unidades de la capa LSTM coincidan con tu entrenamiento original.")