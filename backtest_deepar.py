import os

# --- FORZAR MODO COMPATIBILIDAD ---
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

# --- CONFIGURACIÓN SINCRONIZADA ---
SYMBOL = "TSLA"
WINDOW_SIZE = 100  # <--- Sincronizado con tu entrenamiento de 40 epochs
STEPS_TO_BACKTEST = 12  # 1 hora
LOOKBACK_OFFSET = 20  # Retrocedemos 20 velas (100 min) para probar la IA
MODEL_PATH = f"models/{SYMBOL}/deepar_model.keras"
SCALER_PATH = f"models/{SYMBOL}/scaler.gz"


def run_backtest():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: No existe el modelo para {SYMBOL}")
        return

    # Parche de compatibilidad indispensable
    custom_objects = {
        "DistributionLambda": tfp.layers.DistributionLambda,
        "tf": tf
    }

    print(f"🧠 Cargando cerebro reforzado de {SYMBOL}...")
    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False,
        custom_objects=custom_objects
    )
    scaler = joblib.load(SCALER_PATH)

    # 1. Obtener datos
    df = yf.download(SYMBOL, period="5d", interval="5m", progress=False).dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Simular el pasado
    total_len = len(df)
    split_point = total_len - LOOKBACK_OFFSET

    # Aseguramos que tenemos suficientes datos para la ventana de 100
    if split_point < WINDOW_SIZE:
        print("⚠️ No hay suficientes velas históricas para este OFFSET. Reduciendo OFFSET...")
        split_point = WINDOW_SIZE

    historical_data = df.iloc[:split_point]
    # Tomamos la realidad para comparar (máximo hasta el final de los datos disponibles)
    real_future = df.iloc[split_point: split_point + STEPS_TO_BACKTEST]

    data_scaled = scaler.transform(historical_data[['Close', 'Volume']].values)
    current_window = data_scaled[-WINDOW_SIZE:]

    # 3. Simulación Monte Carlo
    print(f"🕵️ Simulando predicción desde el punto de corte ({len(real_future)} velas de prueba)...")
    n_scenarios = 100
    all_scenarios = np.zeros((n_scenarios, STEPS_TO_BACKTEST))

    for s in range(n_scenarios):
        temp_window = current_window.copy()
        for step in range(STEPS_TO_BACKTEST):
            input_data = np.expand_dims(temp_window, axis=0)
            dist = model(input_data)
            sample_pred = dist.sample().numpy().flatten()[0]
            all_scenarios[s, step] = sample_pred

            # Alimentar el siguiente paso
            new_row = np.array([sample_pred, temp_window[-1, 1]])
            temp_window = np.vstack([temp_window[1:], new_row])

    # 4. Procesar resultados (Des-escalar)
    mean_path = np.mean(all_scenarios, axis=0)
    std_path = np.std(all_scenarios, axis=0)

    def descale(arr):
        d = np.zeros((len(arr), 2))
        d[:, 0] = arr
        return scaler.inverse_transform(d)[:, 0]

    future_mean = descale(mean_path)
    future_upper = descale(mean_path + 2 * std_path)
    future_lower = descale(mean_path - 2 * std_path)

    # 5. Gráfica Comparativa
    plt.figure(figsize=(13, 6))

    # Lo que la IA conocía (últimas 30 velas antes del corte)
    hist_to_plot = historical_data['Close'].tail(30).values
    x_hist = np.arange(len(hist_to_plot))
    plt.plot(x_hist, hist_to_plot, label="Pasado (Datos de entrada)", color="black", alpha=0.7)

    # La Realidad
    x_real = np.arange(len(hist_to_plot) - 1, len(hist_to_plot) - 1 + len(real_future) + 1)
    y_real = np.concatenate([[hist_to_plot[-1]], real_future['Close'].values])
    plt.plot(x_real, y_real, label="REALIDAD", color="#00ff00", lw=3, marker='o', markersize=4)

    # La Predicción
    x_pred = np.arange(len(hist_to_plot) - 1, len(hist_to_plot) - 1 + STEPS_TO_BACKTEST + 1)
    y_pred = np.concatenate([[hist_to_plot[-1]], future_mean])
    plt.plot(x_pred, y_pred, '--r', label="Predicción DeepAR", lw=2)

    # Nube de confianza
    y_upper_plot = np.concatenate([[hist_to_plot[-1]], future_upper])
    y_lower_plot = np.concatenate([[hist_to_plot[-1]], future_lower])
    plt.fill_between(x_pred, y_lower_plot, y_upper_plot, color='orange', alpha=0.2, label="Confianza 95%")

    plt.axvline(x=len(hist_to_plot) - 1, color='blue', linestyle=':', label="Punto de simulación")
    plt.title(f"BACKTEST REFORZADO (40 Epochs): {SYMBOL}")
    plt.legend()
    plt.grid(alpha=0.3)

    # 6. Análisis final
    final_real = real_future['Close'].values[-1]
    final_pred = future_mean[-1]

    print(f"\n" + "=" * 50)
    print(f"📊 RESULTADO DEL BACKTEST ({SYMBOL})")
    print(f"Precio Inicial:        ${hist_to_plot[-1]:.2f}")
    print(f"Precio Real Final:     ${final_real:.2f}")
    print(f"Predicción Media:      ${final_pred:.2f}")

    dentro = (final_real <= future_upper[-1] and final_real >= future_lower[-1])
    print(f"¿Atrapado en la nube?: {'✅ SÍ' if dentro else '❌ NO'}")

    # Cálculo de desviación
    desv = abs(final_real - final_pred) / final_real * 100
    print(f"Desviación del objetivo: {desv:.2f}%")
    print("=" * 50)

    plt.show()


if __name__ == "__main__":
    run_backtest()