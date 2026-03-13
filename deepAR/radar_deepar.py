import os
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tf_keras as keras
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.stats as stats
import streamlit as st  # Importamos streamlit para la interfaz móvil

# --- DETECCIÓN DE ENTORNO Y CONFIGURACIÓN ---
# Si detectamos que corre en Streamlit, usamos sus widgets. Si no, usamos tus input()
try:
    # Esto solo funcionará si ejecutas con 'streamlit run'
    st.set_page_config(page_title="Radar DeepAR", layout="centered")
    st.title("📊 Radar de Probabilidades")

    SYMBOL = st.text_input("Ingresa el Simbolo:", value="TSLA").upper().strip()
    niveles_raw = st.text_input("Ingresa niveles interes (separados por coma):", "406.00, 416.38")

    # El botón dispara la ejecución
    ejecutar = st.button("Calcular Probabilidades")
except:
    # Si lo corres normal en tu PC (python radar_deepar.py)
    SYMBOL = input("Ingresa el Simbolo: ").upper().strip()
    niveles_raw = input("Ingresa niveles interes: ")
    ejecutar = True

# Procesamiento de niveles (tu lógica original)
if niveles_raw:
    niveles_interes = [float(n.strip()) for n in niveles_raw.split(",")]
else:
    niveles_interes = []

WINDOW_SIZE = 100
STEPS_TO_FORECAST = 12
MODEL_PATH = f"./deepAR/models/{SYMBOL}/deepAR_model.keras"
SCALER_PATH = f"./deepAR/models/{SYMBOL}/scalerAR.gz"


# 1. FUNCIÓN MATEMÁTICA DE PROBABILIDAD (INTACTA)
def calcular_probabilidad_temporal(dist, target_price, precio_actual, scaler):
    mu_scaled = dist.mean().numpy().flatten()[0]
    sigma_scaled = dist.stddev().numpy().flatten()[0]
    rango_precio = scaler.data_range_[0]
    sigma_real = sigma_scaled * rango_precio
    mu_real = (mu_scaled * rango_precio) + scaler.data_min_[0]
    z_score = (target_price - precio_actual) / (sigma_real * 2.5 + 1e-6)

    if target_price > precio_actual:
        prob = (1 - stats.norm.cdf(z_score)) * 100
    else:
        prob = stats.norm.cdf(z_score) * 100

    eta_velas = int(abs(target_price - precio_actual) / (sigma_real + 1e-9))
    return round(prob, 1), eta_velas


def descale_series(series, scaler):
    series = np.array(series).reshape(-1, 1)
    dummy = np.zeros((len(series), 2))
    dummy[:, 0] = series[:, 0]
    return scaler.inverse_transform(dummy)[:, 0]


def run_radar_deepar():
    custom_objects = {"DistributionLambda": tfp.layers.DistributionLambda, "tf": tf}

    if not os.path.exists(MODEL_PATH):
        msg = f"❌ Error: Modelo no encontrado en {MODEL_PATH}"
        if 'st' in globals():
            st.error(msg)
        else:
            print(msg)
        return

    # Usamos st.write para que lo veas en el móvil, o print para la PC
    log_msg = f"🧠 Cargando modelo DeepAR para {SYMBOL}..."
    if 'st' in globals():
        st.info(log_msg)
    else:
        print(log_msg)

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects=custom_objects,
        safe_mode=False
    )
    scaler = joblib.load(SCALER_PATH)

    df = yf.download(SYMBOL, period="5d", interval="5m", progress=False).dropna()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

    actual_p = df['Close'].iloc[-1]
    data_scaled = scaler.transform(df[['Close', 'Volume']].values)
    current_window = data_scaled[-WINDOW_SIZE:]

    input_data = np.expand_dims(current_window, axis=0)
    dist_pred = model(input_data)

    # PREPARAR SALIDA
    header = f"📊 RADAR DE PROBABILIDAD DEEPAR: {SYMBOL} | PRECIO: ${actual_p:.2f}"

    if 'st' in globals():
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
        print("█" * 70)
        print(f"{'OBJETIVO':<12} | {'DISTANCIA':<10} | {'PROBABILIDAD':<15} | {'ETA EST.'}")
        print("-" * 70)
        for target in niveles_interes:
            prob, eta = calcular_probabilidad_temporal(dist_pred, target, actual_p, scaler)
            dist_pct = ((target - actual_p) / actual_p) * 100
            signo = "+" if target > actual_p else ""
            print(f"${target:<11.2f} | {signo}{dist_pct:>7.2f}% | {prob:>12}% | {eta:>2} velas")
        print("-" * 70)


if __name__ == "__main__":
    if ejecutar:
        run_radar_deepar()