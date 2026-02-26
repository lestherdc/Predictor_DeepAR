import os

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_KERAS'] = '1'

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tf_keras as keras

# --- CONFIGURACIÓN ---
SYMBOL = "TSLA"
WINDOW_SIZE = 60
STEPS_TO_FORECAST = 12
MODEL_PATH = f"models/{SYMBOL}/deepar_model.keras"
SCALER_PATH = f"models/{SYMBOL}/scaler.gz"


def descale_series(series, scaler):
    series = np.array(series).reshape(-1, 1)
    dummy = np.zeros((len(series), 2))
    dummy[:, 0] = series[:, 0]
    return scaler.inverse_transform(dummy)[:, 0]


def run_radar_vision_objetivos():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: No existe modelo para {SYMBOL}")
        return

    model = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    scaler = joblib.load(SCALER_PATH)

    df = yf.download(SYMBOL, period="5d", interval="5m", progress=False).dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    data_scaled = scaler.transform(df[['Close', 'Volume']].values)
    current_window = data_scaled[-WINDOW_SIZE:]

    n_scenarios = 100
    all_scenarios = np.zeros((n_scenarios, STEPS_TO_FORECAST))

    for s in range(n_scenarios):
        temp_window = current_window.copy()
        for step in range(STEPS_TO_FORECAST):
            input_data = np.expand_dims(temp_window, axis=0)
            dist = model(input_data)
            sample_pred = dist.sample().numpy().flatten()[0]
            all_scenarios[s, step] = sample_pred
            new_row = np.array([sample_pred, temp_window[-1, 1]])
            temp_window = np.vstack([temp_window[1:], new_row])

    mean_path = np.mean(all_scenarios, axis=0)
    std_path = np.std(all_scenarios, axis=0)

    future_mean = descale_series(mean_path, scaler)
    future_upper = descale_series(mean_path + std_path, scaler)
    future_lower = descale_series(mean_path - std_path, scaler)
    actual_price = df['Close'].iloc[-1]

    # --- REPORTE DETALLADO ---
    print(f"\n" + "█" * 60)
    print(f" 🎯 ESTRATEGIA MULTI-HORIZONTE DEEPAR - {SYMBOL}")
    print("█" * 60)
    print(f"💰 PRECIO ACTUAL: ${actual_price:.2f}")
    print("-" * 60)

    # 5 MINUTOS
    m5, u5, l5 = future_mean[0], future_upper[0], future_lower[0]
    var5 = ((u5 - l5) / m5) * 100
    print(f"⏱️  PROYECCIÓN 5 MINUTOS (Inmediato)")
    print(f"   • OBJETIVO MEDIO:  ${m5:.2f}")
    print(f"   • RANGO ESPERADO:  ${l5:.2f} - ${u5:.2f}")
    print(f"   • VARIANZA:        {var5:.2f}%")
    print("-" * 40)

    # 15 MINUTOS
    m15, u15, l15 = future_mean[2], future_upper[2], future_lower[2]
    var15 = ((u15 - l15) / m15) * 100
    print(f"⏱️  PROYECCIÓN 15 MINUTOS (Corto Plazo)")
    print(f"   • OBJETIVO MEDIO:  ${m15:.2f}")
    print(f"   • RANGO ESPERADO:  ${l15:.2f} - ${u15:.2f}")
    print(f"   • VARIANZA:        {var15:.2f}%")
    print("-" * 40)

    # 60 MINUTOS
    m60, u60, l60 = future_mean[-1], future_upper[-1], future_lower[-1]
    var60 = ((u60 - l60) / m60) * 100
    print(f"⏱️  PROYECCIÓN 60 MINUTOS (Tendencia Hora)")
    print(f"   • OBJETIVO MEDIO:  ${m60:.2f}")
    print(f"   • RANGO ESPERADO:  ${l60:.2f} - ${u60:.2f}")
    print(f"   • VARIANZA:        {var60:.2f}%")
    print("█" * 60)

    # --- GRÁFICA ---
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'].tail(30).values, label="Histórico", color="#1f77b4", lw=2)
    x_fut = np.arange(29, 29 + STEPS_TO_FORECAST + 1)
    full_mean = np.concatenate([[actual_price], future_mean])
    plt.plot(x_fut, full_mean, color='red', linestyle='--', label="Predicción Central")
    plt.fill_between(x_fut,
                     np.concatenate([[actual_price], future_lower]),
                     np.concatenate([[actual_price], future_upper]),
                     color='orange', alpha=0.2, label="Nube de Incertidumbre")

    # Marcadores de objetivo
    plt.scatter([30, 32, 41], [m5, m15, m60], color='black', zorder=10)
    plt.annotate(f"${m5:.2f}", (30, m5), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    plt.annotate(f"${m15:.2f}", (32, m15), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    plt.annotate(f"${m60:.2f}", (41, m60), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    plt.title(f"Radar de Objetivos DeepAR - {SYMBOL}")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


if __name__ == "__main__":
    run_radar_vision_objetivos()