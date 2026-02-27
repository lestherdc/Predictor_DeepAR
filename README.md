# Predictor DeepAR
Este predictor es un hibrido, el cual usa un motor interno SVJ o mejor conocido como
Stochastic Volatility Jump. Este predictor a diferencia del DeepAR original, solo
calcula probabilidad basandose en la volatilidad actual, pero no sabe proyectar
velas hacia el futuro. Por esta razon no puedo definir un "cuando" llegara 
el precio junto con una probabilidad.


## Estructura del proyecto

---

## 🧠 Descripción General

- **data/** → Almacena los datasets históricos descargados.
- **models/** → Contiene los modelos entrenados y los scalers.
- **src/** → Núcleo del sistema: procesamiento, modelo DeepAR y lógica predictiva.
- **trainer.py** → Entrena modelos para múltiples acciones.
- **main.py** → Ejecuta predicción en tiempo real y visualización.

---


## Radar
A diferencia de otros proyectos, que solo imprimia texto,
este radar_deepar.py deberia ser capaz de 

- Cargar el modelo de probabilidades
- hacer una prediccion Monte Carlo (Como ya vimos es dar multiples futuros posibles)
- Graficar: debemos tener graficas para poder hacer nuestro propio analisis
