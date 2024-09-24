import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np

app = Flask(__name__, static_url_path='')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

stop_training = False

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

@app.route('/train', methods=['POST'])
def train_model():
    global stop_training
    stop_training = False

    data = request.json
    learning_rate = data['learning_rate']
    activation = data['activation']
    regularization = getattr(tf.keras.regularizers, data['regularization'])
    reg_rate = data['reg_rate']
    train_split = data['train_split']

    split_index = int(len(x_train) * train_split)
    x_train_split, y_train_split = x_train[:split_index], y_train[:split_index]
    x_val_split, y_val_split = x_train[split_index:], y_train[split_index:]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_split, y_train_split)).shuffle(buffer_size=1024).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val_split, y_val_split)).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model = create_model(learning_rate, activation, regularization, reg_rate)
    history = {'accuracy': [], 'val_accuracy': []}

    for epoch in range(1, 101):  # Arbitrary large number of epochs
        if stop_training:
            break
        hist = model.fit(train_dataset, epochs=1, validation_data=val_dataset)
        history['accuracy'].extend(hist.history['accuracy'])
        history['val_accuracy'].extend(hist.history['val_accuracy'])

    # Truncate accuracy values to 5 decimal places
    accuracy = [round(acc, 5) for acc in history['accuracy']]
    val_accuracy = [round(val_acc, 5) for val_acc in history['val_accuracy']]

    latest_accuracy = accuracy[-1]
    latest_val_accuracy = val_accuracy[-1]

    return jsonify({'latest_accuracy': latest_accuracy, 'latest_val_accuracy': latest_val_accuracy, 'accuracy': accuracy, 'val_accuracy': val_accuracy})

@app.route('/stop', methods=['POST'])
def stop_training_route():
    global stop_training
    stop_training = True
    return jsonify({'status': 'Training stopped'})

if __name__ == '__main__':
    app.run(debug=True)