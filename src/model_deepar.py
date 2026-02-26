import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def build_deepar_model(window_size, n_features):
    inputs = tf.keras.layers.Input(shape=(window_size, n_features))

    #Capas de memoria profunda
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(16, activation='relu')(x)

    #Salida: Media (mu) y desciacion (sigma)
    #Usamos 2 neuronas para definir la campa de Gauss
    params = tf.keras.layers.Dense(2)(x)

    #Convertimos los parametros en una distribucion normal real
    dist_output = tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(
            loc=t[..., :1],
            scale=1e-3 + tf.math.softplus(t[..., 1:]) #Asegura que la desviacion sea positiva
        )
    )(params)

    model = tf.keras.Model(inputs=inputs, outputs=dist_output)

    #Perdida: Log-Likehood Negativo (Esto medira que tan bien la curva contiene los datos)
    def negative_log_likelihood(y_true, dist):
        return -dist.log_prob(y_true)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=negative_log_likelihood)
    return model

