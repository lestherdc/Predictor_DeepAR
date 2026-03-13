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


# --- CONFIGURACIÓN DE DISTRIBUCIÓN (LA PIEZA QUE FALTABA) ---
# Esta función es necesaria para que load_model sepa reconstruir la capa de DeepAR
def model_dist(t):
    return tfp.distributions.Normal(
        loc=t[..., :1],
        scale=1e-3 + tf.math.softplus(t[..., 1:])
    )


# --- DETECCIÓN DE ENTORNO ---
try:
    if st.runtime.exists():
        st.set_page_config(page_title="Radar DeepAR", layout="centered")
        st.title("📊 Radar de Probabilidades")
        SYMBOL = st.text_input("Ingresa el Simbolo:", value="TSLA").upper().strip()
        niveles_raw = st.text_input("Ingresa niveles interes (separados por coma):", "406.00, 416.38")
        ejecutar = st.button("Calcular Probabilidades")
    else:
        raise Exception("Not in streamlit")
except:
    SYMBOL = input("Ingresa el Simbolo: ").upper().strip()
    niveles_raw = input("Ingresa niveles interes: ")
    ejecutar = True

if niveles_raw:
    niveles_interes = [float(n.strip()) for n in niveles_raw.split(",")]
else:
    niveles_interes = []

WINDOW_SIZE = 100
STEPS_TO_FORECAST = 12
# Ajuste de ruta para compatibilidad con Streamlit Cloud
MODEL_PATH = f"models/{SYMBOL}/deepAR_model.keras"
SCALER_PATH = f"models/{SYMBOL}/scalerAR.gz"


def calcular_probabilidad_temporal(dist, target_price, precio_actual, scaler):
    # Extraemos media y desviación de la distribución predicha
    mu_scaled = dist.mean().numpy().flatten()[0]
    sigma_scaled = dist.stddev().numpy().flatten()[0]

    # El scaler de 2 columnas (Close, Volume)
    rango_precio = scaler.data_range_[0]
    mu_real = (mu_scaled * rango_precio) + scaler.data_min_[0]
    sigma_real = sigma_scaled * rango_precio

    # Cálculo de Z-Score basado en tu volatilidad institucional
    z_score = (target_price - precio_actual) / (sigma_real * 2.5 + 1e-6)

    if target_price > precio_actual:
        prob = (1 - stats.norm.cdf(z_score)) * 100
    else:
        prob = stats.norm.cdf(z_score) * 100

    eta_velas = int(abs(target_price - precio_actual) / (sigma_real + 1e-9))
    return round(prob, 1), eta_velas


def run_radar_deepar():
    # SOLUCIÓN AL TYPEERROR: Pasamos la función y el módulo a custom_objects
    custom_objects = {
        "DistributionLambda": tfp.layers.DistributionLambda,
        "model_dist": model_dist,  # Agregamos la referencia a la función
        "tf": tf
    }

    if not os.path.exists(MODEL_PATH):
        msg = f"❌ Error: Modelo no encontrado en {MODEL_PATH}"
        if 'st' in globals() and st.runtime.exists():
            st.error(msg)
        else:
            print(msg)
        return

    log_msg = f"🧠 Cargando modelo DeepAR para {SYMBOL}..."
    if 'st' in globals() and st.runtime.exists():
        st.info(log_msg)
    else:
        print(log_msg)

    # Carga robusta
    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects=custom_objects
    )
    scaler = joblib.load(SCALER_PATH)

    # Descarga de datos
    df = yf.download(SYMBOL, period="5d", interval="5m", progress=False).dropna()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

    actual_p = df['Close'].iloc[-1]
    data_scaled = scaler.transform(df[['Close', 'Volume']].values)
    current_window = data_scaled[-WINDOW_SIZE:]

    input_data = np.expand_dims(current_window, axis=0)
    # Ejecutamos el modelo para obtener la distribución
    dist_pred = model(input_data)

    header = f"📊 RADAR DE PROBABILIDAD DEEPAR: {SYMBOL} | PRECIO: ${actual_p:.2f}"

    if 'st' in globals() and st.runtime.exists():
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
        print("\n" + "█" * 70)
        print(header)
        # ... (resto de tu print igual)
        for target in niveles_interes:
            prob, eta = calcular_probabilidad_temporal(dist_pred, target, actual_p, scaler)
            dist_pct = ((target - actual_p) / actual_p) * 100
            signo = "+" if target > actual_p else ""
            print(f"${target:<11.2f} | {signo}{dist_pct:>7.2f}% | {prob:>12}% | {eta:>2} velas")


if __name__ == "__main__":
    if ejecutar:
        run_radar_deepar()