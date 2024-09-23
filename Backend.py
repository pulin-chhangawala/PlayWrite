from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np

app = Flask(__name__)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model(learning_rate, activation, regularization, reg_rate):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=activation, kernel_regularizer=regularization(reg_rate)),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

@app.route('/')

def serve_index():
    return send_from_directory('.', 'index.html')

def train_model():
    data = request.json
    learning_rate = data['learning_rate']
    activation = data['activation']
    regularization = getattr(tf.keras.regularizers, data['regularization'])
    reg_rate = data['reg_rate']
    train_split = data['train_split']

    split_index = int(len(x_train) * train_split)
    x_train_split, y_train_split = x_train[:split_index], y_train[:split_index]
    x_val_split, y_val_split = x_train[split_index:], y_train[split_index:]

    model = create_model(learning_rate, activation, regularization, reg_rate)
    history = model.fit(x_train_split, y_train_split, epochs=5, validation_data=(x_val_split, y_val_split))

    return jsonify({'accuracy': history.history['accuracy'], 'val_accuracy': history.history['val_accuracy']})

if __name__ == '__main__':
    app.run(debug=True)