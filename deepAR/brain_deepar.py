import tensorflow as tf
from tf_keras import layers, models
import tensorflow_probability as tfp


class DeepARModel:
    def __init__(self, window_size, feature_count):
        self.window_size = window_size
        self.feature_count = feature_count

    def build_model(self):
        inputs = layers.Input(shape=(self.window_size, self.feature_count))

        # Capas de memoria (LSTM)
        x = layers.LSTM(64, return_sequences=True)(inputs)
        x = layers.Dropout(0.1)(x)
        x = layers.LSTM(32)(x)

        # Salida: Media y Volatilidad (mu y sigma)
        params = layers.Dense(2)(x)

        # Capa Probabilística de DeepAR
        dist_layer = tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Normal(
                loc=t[..., :1],
                scale=1e-3 + tf.math.softplus(t[..., 1:])
            )
        )(params)

        model = models.Model(inputs=inputs, outputs=dist_layer)

        # Función de pérdida: Negative Log-Likelihood
        def nll(y_true, dist):
            return -dist.log_prob(y_true)

        model.compile(optimizer='adam', loss=nll)
        return model