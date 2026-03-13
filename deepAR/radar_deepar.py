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

# --- 1. CONFIGURACIÓN DE RUTAS ABSOLUTAS ---
# Obtenemos la ruta del directorio donde reside este archivo .py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --- 2. DEFINICIÓN DE LA DISTRIBUCIÓN (PARA CARGA DE MODELO) ---
def model_dist(t):
    return tfp.distributions.Normal(
        loc=t[..., :1],
        scale=1e-3 + tf.math.softplus(t[..., 1:])
    )


# --- 3. INTERFAZ Y ENTRADA DE DATOS ---
try:
    if st.runtime.exists():
        st.set_page_config(page_title="Radar DeepAR", layout="centered")
        st.title("📊 Radar de Probabilidades")
        SYMBOL = st.text_input("Ingresa el Simbolo:", value="TSLA").upper().strip()
        niveles_raw = st.text_input("Ingresa niveles interés (separados por coma):", "406.00, 416.38")
        ejecutar = st.button("Calcular Probabilidades")
    else:
        raise Exception("Ejecución fuera de Streamlit")
except Exception:
    SYMBOL = input("Ingresa el Simbolo: ").upper().strip()
    niveles_raw = input("Ingresa niveles interés: ")
    ejecutar = True

# Procesamiento de niveles
if niveles_raw:
    niveles_interes = [float(n.strip()) for n in niveles_raw.split(",")]
else:
    niveles_interes = []

WINDOW_SIZE = 100

# --- 4. CONSTRUCCIÓN DE RUTAS DINÁMICAS ---
# Esto evita el error de "Modelo no encontrado"
MODEL_PATH = os.path.join(BASE_DIR, "models", SYMBOL, "deepAR_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", SYMBOL, "scalerAR.gz")


# --- 5. LÓGICA MATEMÁTICA ---
def calcular_probabilidad_temporal(dist, target_price, precio_actual, scaler):
    mu_scaled = dist.mean().numpy().flatten()[0]
    sigma_scaled = dist.stddev().numpy().flatten()[0]

    # El scaler fue entrenado con [Close, Volume], usamos el rango del Close (índice 0)
    rango_precio = scaler.data_range_[0]
    mu_real = (mu_scaled * rango_precio) + scaler.data_min_[0]
    sigma_real = sigma_scaled * rango_precio

    # Ajuste de Z-Score para volatilidad institucional (2.5 sigmas)
    z_score = (target_price - precio_actual) / (sigma_real * 2.5 + 1e-6)

    if target_price > precio_actual:
        prob = (1 - stats.norm.cdf(z_score)) * 100
    else:
        prob = stats.norm.cdf(z_score) * 100

    eta_velas = int(abs(target_price - precio_actual) / (sigma_real + 1e-9))
    return round(prob, 1), eta_velas


# --- 6. EJECUCIÓN PRINCIPAL ---
def run_radar_deepar():
    # Objetos personalizados para reconstruir la capa Probabilística
    custom_objects = {
        "DistributionLambda": tfp.layers.DistributionLambda,
        "model_dist": model_dist,
        "tf": tf
    }

    # Verificación de existencia de archivos
    if not os.path.exists(MODEL_PATH):
        msg = f"❌ Error: Archivo no encontrado en {MODEL_PATH}"
        if 'st' in globals() and st.runtime.exists():
            st.error(msg)
        else:
            print(msg)
        return

    log_msg = f"🧠 Analizando estructura de {SYMBOL}..."
    if 'st' in globals() and st.runtime.exists():
        st.info(log_msg)
    else:
        print(log_msg)

    try:
        # Carga del modelo y el scaler
        model = keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects=custom_objects
        )
        scaler = joblib.load(SCALER_PATH)

        # Obtención de datos actuales
        df = yf.download(SYMBOL, period="5d", interval="5m", progress=False).dropna()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        actual_p = df['Close'].iloc[-1]

        # Preprocesamiento de ventana
        data_scaled = scaler.transform(df[['Close', 'Volume']].values)
        current_window = data_scaled[-WINDOW_SIZE:]
        input_data = np.expand_dims(current_window, axis=0)

        # Inferencia (DeepAR devuelve la distribución completa)
        dist_pred = model(input_data)

        header = f"📊 RADAR DE PROBABILIDAD: {SYMBOL} | PRECIO: ${actual_p:.2f}"

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
            print(f"\n{header}\n" + "-" * 50)
            for target in niveles_interes:
                prob, eta = calcular_probabilidad_temporal(dist_pred, target, actual_p, scaler)
                print(f"Target: ${target:.2f} | Prob: {prob}% | ETA: {eta} velas")

    except Exception as e:
        error_msg = f"🔥 Error en ejecución: {str(e)}"
        if 'st' in globals() and st.runtime.exists():
            st.error(error_msg)
        else:
            print(error_msg)


if __name__ == "__main__":
    if ejecutar:
        run_radar_deepar()