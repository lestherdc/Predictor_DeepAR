import numpy as np
from sklearn.preprocessing import RobustScaler

class DeepARDDataEngine:
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.scaler = RobustScaler()

    def prepare_data(self, df, training = True):
        # Usaremos el close del valor y el volumen como base de la prediccion.
        data = df[['Close', 'Volume']].values

        if training:
            scaled_data = self.scaler.fit_transform(data)
        else:
            scaled_data = self.scaler.transform(data)

        X, y = [], []
        for i in range(len(scaled_data) - self.window_size):
            X.append(scaled_data[i:i+self.window_size])
            #DeepAR predice la distribucion del siguiente valor
            y.append(scaled_data[i+self.window_size, 0])

        return np.array(X), np.array(y)

    def get_scaler(self):
        return self.scaler