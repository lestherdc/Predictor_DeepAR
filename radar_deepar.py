import os

# --- FORZAR MODO COMPATIBILIDAD KERAS 2/3 ---
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_KERAS'] = '1'

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tf_keras as keras
import tensorflow as tf
import tensorflow_probability as tfp

# --- CONFIGURACIÓN REFORZADA ---
SYMBOL = "PLTR"
WINDOW_SIZE = 100  # Ajustado a tu nuevo entrenamiento
STEPS_TO_FORECAST = 12
MODEL_PATH = f"models/{SYMBOL}/deepar_model.keras"
SCALER_PATH = f"models/{SYMBOL}/scaler.gz"


def descale_series(series, scaler):
    """Convierte datos normalizados a precios USD reales"""
    series = np.array(series).reshape(-1, 1)
    dummy = np.zeros((len(series), 2))
    dummy[:, 0] = series[:, 0]
    return scaler.inverse_transform(dummy)[:, 0]


def run_radar_completo():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: El modelo de {SYMBOL} no existe en la carpeta models/.")
        return

    # Parche de objetos personalizados para cargar el modelo de probabilidad
    custom_objects = {
        "DistributionLambda": tfp.layers.DistributionLambda,
        "tf": tf
    }

    print(f"🧠 Cargando modelo de {SYMBOL} (Optimizado: 40 Epochs)...")
    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False,
        custom_objects=custom_objects
    )
    scaler = joblib.load(SCALER_PATH)

    # 1. Obtención de datos recientes
    df = yf.download(SYMBOL, period="5d", interval="5m", progress=False).dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    data_scaled = scaler.transform(df[['Close', 'Volume']].values)

    if len(data_scaled) < WINDOW_SIZE:
        print(f"⚠️ Datos insuficientes: Necesitas {WINDOW_SIZE} velas, tienes {len(data_scaled)}.")
        return

    current_window = data_scaled[-WINDOW_SIZE:]

    # 2. Simulación de Monte Carlo (100 escenarios posibles)
    print(f"🔮 Proyectando rutas probabilísticas para la próxima hora...")
    n_scenarios = 100
    all_scenarios = np.zeros((n_scenarios, STEPS_TO_FORECAST))

    for s in range(n_scenarios):
        temp_window = current_window.copy()
        for step in range(STEPS_TO_FORECAST):
            input_data = np.expand_dims(temp_window, axis=0)
            dist = model(input_data)
            sample_pred = dist.sample().numpy().flatten()[0]
            all_scenarios[s, step] = sample_pred

            # Re-alimentar la predicción al modelo
            new_row = np.array([sample_pred, temp_window[-1, 1]])
            temp_window = np.vstack([temp_window[1:], new_row])

    # 3. Procesamiento Estadístico
    mean_path = np.mean(all_scenarios, axis=0)
    std_path = np.std(all_scenarios, axis=0)

    # Conversión a USD
    f_mean = descale_series(mean_path, scaler)
    f_upper = descale_series(mean_path + 2 * std_path, scaler)  # Banda de confianza 95%
    f_lower = descale_series(mean_path - 2 * std_path, scaler)
    actual_p = df['Close'].iloc[-1]

    # --- REPORTE MULTI-HORIZONTE CON OBJETIVOS Y RANGOS ---
    print(f"\n" + "█" * 65)
    print(f" 📊 RADAR DE PRECISIÓN DEEPAR (40 EPOCHS) - {SYMBOL}")
    print("█" * 65)
    print(f"💰 PRECIO ACTUAL: ${actual_p:.2f}")
    print("-" * 65)

    # Datos 5 MINUTOS (Vela inmediata)
    m5, u5, l5 = f_mean[0], f_upper[0], f_lower[0]
    v5 = ((u5 - l5) / m5) * 100
    print(f"⏱️  A 5 MINUTOS (Próxima Vela):")
    print(f"   • Objetivo: ${m5:.2f}")
    print(f"   • RANGO:    ${l5:.2f} - ${u5:.2f}")
    print(f"   • Varianza: {v5:.2f}%")

    # Datos 15 MINUTOS (Corto plazo)
    m15, u15, l15 = f_mean[2], f_upper[2], f_lower[2]
    v15 = ((u15 - l15) / m15) * 100
    print(f"\n⏱️  A 15 MINUTOS (Tendencia 15m):")
    print(f"   • Objetivo: ${m15:.2f}")
    print(f"   • RANGO:    ${l15:.2f} - ${u15:.2f}")
    print(f"   • Varianza: {v15:.2f}%")

    # Datos 60 MINUTOS (Tendencia 1h)
    m60, u60, l60 = f_mean[-1], f_upper[-1], f_lower[-1]
    v60 = ((u60 - l60) / m60) * 100
    print(f"\n⏱️  A 60 MINUTOS (Cierre Hora):")
    print(f"   • Objetivo: ${m60:.2f}")
    print(f"   • RANGO:    ${l60:.2f} - ${u60:.2f}")
    print(f"   • Varianza: {v60:.2f}%")
    print("█" * 65)

    # 4. Visualización Gráfica
    plt.figure(figsize=(14, 7))

    # Histórico (Velas pasadas)
    hist_vals = df['Close'].tail(40).values
    plt.plot(hist_vals, label="Precio Histórico", color="#1f77b4", lw=2)

    # Proyección (Eje X futuro)
    x_fut = np.arange(len(hist_vals) - 1, len(hist_vals) + STEPS_TO_FORECAST)
    y_mean = np.concatenate([[actual_p], f_mean])
    y_upper = np.concatenate([[actual_p], f_upper])
    y_lower = np.concatenate([[actual_p], f_lower])

    plt.plot(x_fut, y_mean, '--r', label="Predicción Media IA", lw=2)
    plt.fill_between(x_fut, y_lower, y_upper, color='orange', alpha=0.2, label="Nube de Probabilidad (95%)")

    # Marcadores de Puntos Clave
    plt.scatter([len(hist_vals), len(hist_vals) + 2, len(hist_vals) + 11],
                [m5, m15, m60], color='black', zorder=5)

    plt.axvline(x=len(hist_vals) - 1, color='gray', linestyle=':', alpha=0.5)
    plt.title(f"Visión 360° DeepAR: {SYMBOL} | Multi-Horizonte Proyectado", fontsize=14)
    plt.ylabel("Precio USD")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.show()


if __name__ == "__main__":
    run_radar_completo()