import os
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tf_keras as keras
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.stats as stats
import streamlit as st

# --- 1. CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --- 2. FUNCIÓN DE DISTRIBUCIÓN (BLINDADA) ---
def model_dist(t):
    """
    Esta función reconstruye la salida probabilística.
    Usamos tf.cast y convert_to_tensor para limpiar cualquier residuo de Numpy.
    """
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    # Separamos media y desviación estándar
    loc = t[..., :1]
    scale = 1e-3 + tf.math.softplus(t[..., 1:])
    return tfp.distributions.Normal(loc=loc, scale=scale)


# --- 3. INTERFAZ DE USUARIO ---
try:
    if st.runtime.exists():
        st.set_page_config(page_title="Radar DeepAR", layout="centered")
        st.title("📊 Radar de Probabilidades")
        SYMBOL = st.text_input("Ingresa el Símbolo (ej: PLTR, TSLA):", value="TSLA").upper().strip()
        niveles_raw = st.text_input("Niveles de interés (separados por coma):", "406.00, 416.38")
        ejecutar = st.button("Calcular Probabilidades")
    else:
        raise Exception()
except:
    SYMBOL = input("Símbolo: ").upper().strip()
    niveles_raw = input("Niveles: ")
    ejecutar = True

niveles_interes = [float(n.strip()) for n in niveles_raw.split(",")] if niveles_raw else []
WINDOW_SIZE = 100
MODEL_PATH = os.path.join(BASE_DIR, "models", SYMBOL, "deepAR_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", SYMBOL, "scalerAR.gz")


# --- 4. CÁLCULO DE PROBABILIDAD ---
def calcular_probabilidad_temporal(dist, target_price, precio_actual, scaler):
    # Obtenemos media y desviación desde el objeto de distribución de TFP
    mu_scaled = tf.reduce_mean(dist.mean()).numpy()
    sigma_scaled = tf.reduce_mean(dist.stddev()).numpy()

    rango_precio = scaler.data_range_[0]
    mu_real = (mu_scaled * rango_precio) + scaler.data_min_[0]
    sigma_real = sigma_scaled * rango_precio

    # Z-Score ajustado a 2.5 sigmas (volatilidad institucional)
    z_score = (target_price - precio_actual) / (sigma_real * 2.5 + 1e-6)

    if target_price > precio_actual:
        prob = (1 - stats.norm.cdf(z_score)) * 100
    else:
        prob = stats.norm.cdf(z_score) * 100

    eta_velas = int(abs(target_price - precio_actual) / (sigma_real + 1e-9))
    return round(float(prob), 1), eta_velas


# --- 5. EJECUCIÓN ---
def run_radar_deepar():
    # Mapeamos la capa y la función para que Keras sepa cómo reconstruirlas
    custom_objects = {
        "DistributionLambda": tfp.layers.DistributionLambda,
        "model_dist": model_dist,
        "tf": tf
    }

    if not os.path.exists(MODEL_PATH):
        st.error(f"Modelo no encontrado en: {MODEL_PATH}")
        return

    try:
        # Cargamos el modelo ignorando la seguridad de Lambda y forzando objetos personalizados
        model = keras.models.load_model(
            MODEL_PATH,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False
        )
        scaler = joblib.load(SCALER_PATH)

        # Datos de mercado
        df = yf.download(SYMBOL, period="5d", interval="5m", progress=False).dropna()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        actual_p = df['Close'].iloc[-1]
        data_scaled = scaler.transform(df[['Close', 'Volume']].values)
        current_window = data_scaled[-WINDOW_SIZE:]
        input_data = np.expand_dims(current_window, axis=0).astype(np.float32)

        # Inferencia
        dist_pred = model(input_data)

        header = f"📊 RADAR {SYMBOL} | PRECIO ACTUAL: ${actual_p:.2f}"

        if st.runtime.exists():
            st.markdown(f"### {header}")
            resultados = []
            for target in niveles_interes:
                prob, eta = calcular_probabilidad_temporal(dist_pred, target, actual_p, scaler)
                dist_pct = ((target - actual_p) / actual_p) * 100
                signo = "+" if target > actual_p else ""
                resultados.append({
                    "OBJETIVO": f"${target:.2f}",
                    "DISTANCIA": f"{signo}{dist_pct:.2f}%",
                    "PROBABILIDAD": f"{prob}%",
                    "ETA": f"{eta} velas"
                })
            st.table(pd.DataFrame(resultados))
        else:
            print(f"\n{header}\n" + "-" * 50)
            for target in niveles_interes:
                prob, eta = calcular_probabilidad_temporal(dist_pred, target, actual_p, scaler)
                print(f"Target: ${target:.2f} | Prob: {prob}% | ETA: {eta} velas")

    except Exception as e:
        st.error(f"🔥 Error Crítico: {str(e)}")


if __name__ == "__main__":
    if ejecutar:
        run_radar_deepar()