import os
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tf_keras as keras
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.stats as stats

# --- CONFIGURACIÓN ---
SYMBOL = "TSLA"
WINDOW_SIZE = 100
STEPS_TO_FORECAST = 12
MODEL_PATH = f"models/{SYMBOL}/deepAR_model.keras"
SCALER_PATH = f"models/{SYMBOL}/scalerAR.gz"


# 1. FUNCIÓN MATEMÁTICA DE PROBABILIDAD
def calcular_probabilidad_temporal(dist, target_price, precio_actual, scaler):
    # 1. Extraemos mu y sigma (valores escalados entre 0 y 1)
    mu_scaled = dist.mean().numpy().flatten()[0]
    sigma_scaled = dist.stddev().numpy().flatten()[0]

    # 2. DESESCALAMOS la volatilidad para llevarla a DÓLARES reales
    # La volatilidad real es (sigma_escalada * rango_del_scaler)
    rango_precio = scaler.data_range_[0]  # La diferencia entre el max y min del entrenamiento
    sigma_real = sigma_scaled * rango_precio
    mu_real = (mu_scaled * rango_precio) + scaler.data_min_[0]

    # 3. Cálculo del Z-Score con valores REALES en dólares
    # Añadimos un multiplicador de sensibilidad (2.5) para captar colas de mercado
    z_score = (target_price - precio_actual) / (sigma_real * 2.5 + 1e-6)

    if target_price > precio_actual:
        prob = (1 - stats.norm.cdf(z_score)) * 100
    else:
        prob = stats.norm.cdf(z_score) * 100

    # ETA corregido con volatilidad real
    eta_velas = int(abs(target_price - precio_actual) / (sigma_real + 1e-9))

    return round(prob, 1), eta_velas

def descale_series(series, scaler):
    series = np.array(series).reshape(-1, 1)
    dummy = np.zeros((len(series), 2))
    dummy[:, 0] = series[:, 0]
    return scaler.inverse_transform(dummy)[:, 0]


def run_radar_deepar():
    # Parche de objetos personalizados para TensorFlow Probability
    custom_objects = {"DistributionLambda": tfp.layers.DistributionLambda, "tf": tf}

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Modelo no encontrado en {MODEL_PATH}")
        return

    print(f"🧠 Cargando modelo DeepAR desde {MODEL_PATH}...")

    # 2. Cargar con safe_mode=False para permitir la capa Lambda
    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects=custom_objects,
        safe_mode=False  # <--- ESTA ES LA CLAVE PARA ELIMINAR EL ERROR
    )
    scaler = joblib.load(SCALER_PATH)

    # Obtener datos recientes
    df = yf.download(SYMBOL, period="5d", interval="5m", progress=False).dropna()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

    actual_p = df['Close'].iloc[-1]
    data_scaled = scaler.transform(df[['Close', 'Volume']].values)
    current_window = data_scaled[-WINDOW_SIZE:]

    # --- PREDICCIÓN PROBABILÍSTICA ---
    input_data = np.expand_dims(current_window, axis=0)
    dist_pred = model(input_data)  # Esto genera la distribución mu/sigma

    # --- NIVELES A TESTEAR (Ejemplos basados en tu interés de visión) ---
    # Aquí es donde pondrás tus niveles detectados de otros días
    niveles_interes = [407.00, 394.10, 387.60, 383.80]

    print(f"\n" + "█" * 70)
    print(f" 📊 RADAR DE PROBABILIDAD DEEPAR: {SYMBOL}")
    print(f" 💰 PRECIO ACTUAL: ${actual_p:.2f}")
    print("█" * 70)

    print(f"{'OBJETIVO':<12} | {'DISTANCIA':<10} | {'PROBABILIDAD':<15} | {'ETA EST.'}")
    print("-" * 70)

    for target in niveles_interes:
        prob, eta = calcular_probabilidad_temporal(dist_pred, target, actual_p, scaler)
        dist_pct = ((target - actual_p) / actual_p) * 100

        # Formateo de salida
        signo = "+" if target > actual_p else ""
        print(f"${target:<11.2f} | {signo}{dist_pct:>7.2f}% | {prob:>12}% | {eta:>2} velas")

    print("-" * 70)
    print(f"INFO: Probabilidades calculadas mediante Z-Score sobre Distribución Normal.")
    print("█" * 70)


if __name__ == "__main__":
    run_radar_deepar()