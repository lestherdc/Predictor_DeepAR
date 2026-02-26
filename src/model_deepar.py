import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras  # IMPORTANTE: Usar tf_keras
from tf_keras import layers

tfd = tfp.distributions


def build_deepar_model(window_size, n_features):
    # Usamos la API de tf_keras explícitamente
    inputs = layers.Input(shape=(window_size, n_features))

    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(32)(x)
    x = layers.Dense(16, activation='relu')(x)

    # Esta capa produce los parámetros brutos
    params = layers.Dense(2)(x)

    # Usamos DistributionLambda con una función que TFP entienda
    dist_output = tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(
            loc=t[..., :1],
            scale=1e-3 + tf.math.softplus(t[..., 1:])
        )
    )(params)

    model = keras.Model(inputs=inputs, outputs=dist_output)

    def negative_log_likelihood(y_true, dist):
        return -dist.log_prob(y_true)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=negative_log_likelihood
    )

    return model