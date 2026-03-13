import os
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tf_keras as keras
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.stats as stats

# --- 1. CONFIGURACIÓN DE RUTAS ABSOLUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --- 2. DEFINICIÓN DE LA DISTRIBUCIÓN (CORREGIDA PARA EVITAR ERRORES DE NUMPY) ---
def model_dist(t):
    # Forzamos la conversión a tensores para evitar conflictos de dimensiones con Numpy
    loc = tf.convert_to_tensor(t[..., :1], dtype=tf.float32)
    scale = 1e-3 + tf.math.softplus(tf.convert_to_tensor(t[..., 1:], dtype=tf.float32))
    return tfp.distributions.Normal(loc=loc, scale=scale)


# --- 3. INTERFAZ Y ENTRADA DE DATOS ---
try:
    if st.runtime.exists():
        st.set_page_config(page_title="Radar DeepAR", layout="centered")
        st.title("📊 Radar de Probabilidades")
        SYMBOL = st.text_input("Ingresa el Simbolo:", value="TSLA").upper().strip()
        niveles_raw = st.text_input("Ingresa niveles interés (separados por coma):", "406.00, 416.38")
        ejecutar = st.button("Calcular Probabilidades")
    else:
        raise Exception("Ejecución local")
except Exception:
    SYMBOL = input("Ingresa el Simbolo: ").upper().strip()
    niveles_raw = input("Ingresa niveles interés: ")
    ejecutar = True

if niveles_raw:
    niveles_interes = [float(n.strip()) for n in niveles_raw.split(",")]
else:
    niveles_interes = []

WINDOW_SIZE = 100
MODEL_PATH = os.path.join(BASE_DIR, "models", SYMBOL, "deepAR_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", SYMBOL, "scalerAR.gz")


# --- 4. LÓGICA DE PROBABILIDAD ---
def calcular_probabilidad_temporal(dist, target_price, precio_actual, scaler):
    mu_scaled = dist.mean().numpy().flatten()[0]
    sigma_scaled = dist.stddev().numpy().flatten()[0]

    rango_precio = scaler.data_range_[0]
    sigma_real = sigma_scaled * rango_precio

    # Z-Score ajustado a volatilidad institucional
    z_score = (target_price - precio_actual) / (sigma_real * 2.5 + 1e-6)

    if target_price > precio_actual:
        prob = (1 - stats.norm.cdf(z_score)) * 100
    else:
        prob = stats.norm.cdf(z_score) * 100

    eta_velas = int(abs(target_price - precio_actual) / (sigma_real + 1e-9))
    return round(prob, 1), eta_velas


# --- 5. EJECUCIÓN PRINCIPAL ---
def run_radar_deepar():
    custom_objects = {
        "DistributionLambda": tfp.layers.DistributionLambda,
        "model_dist": model_dist,
        "tf": tf
    }

    if not os.path.exists(MODEL_PATH):
        msg = f"❌ Error: Modelo no encontrado en {MODEL_PATH}"
        if 'st' in globals() and st.runtime.exists():
            st.error(msg)
        else:
            print(msg)
        return

    try:
        # CARGA CON SEGURIDAD DESACTIVADA (safe_mode=False)
        model = keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects=custom_objects,
            safe_mode=False
        )
        scaler = joblib.load(SCALER_PATH)

        # Obtención de datos reales
        df = yf.download(SYMBOL, period="5d", interval="5m", progress=False).dropna()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        actual_p = df['Close'].iloc[-1]
        data_scaled = scaler.transform(df[['Close', 'Volume']].values)
        current_window = data_scaled[-WINDOW_SIZE:]
        input_data = np.expand_dims(current_window, axis=0)

        # Inferencia
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
            print(f"\n{header}\n" + "-" * 60)
            for target in niveles_interes:
                prob, eta = calcular_probabilidad_temporal(dist_pred, target, actual_p, scaler)
                print(f"Target: ${target:<8.2f} | Prob: {prob:>5}% | ETA: {eta:>2} velas")

    except Exception as e:
        error_msg = f"🔥 Error en ejecución: {str(e)}"
        if 'st' in globals() and st.runtime.exists():
            st.error(error_msg)
        else:
            print(error_msg)


if __name__ == "__main__":
    if ejecutar:
        run_radar_deepar()