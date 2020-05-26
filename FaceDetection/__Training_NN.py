import tensorflow as tf


class FaceNN:

    def __init__(self, input_shape, output_shape):
        self.NN = FaceNN.init_NN(input_shape, output_shape)

    @staticmethod
    def init_NN(input_shape, output_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=48, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(units=24, activation='relu'),
            tf.keras.layers.Dense(units=12, activation='relu'),
            tf.keras.layers.Dense(units=output_shape, activation='softmax'),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005), loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model

    def training(self, train_X, train_Y, save_location, epoch=30, validation_split=0.25):
        self.history = self.NN.fit(train_X, train_Y, epochs=200, batch_size=32, validation_split=0.25)
        self.NN.save_weights(save_location)

    def evaluate_NN(self, test_X, test_Y, weight_location):
        self.NN.load_weights(weight_location)
        self.NN.evaluate(test_X, test_Y)

    def predict_data(self, data, weight_location):
        self.NN.load_weights(weight_location)
        # print("load weights complete")
        result = self.NN.predict(data)
        # print(result[0], result[0].argmax())
        return result.argmax()
