import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import time
import threading
import io
import csv
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Reshape
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

app = Flask(__name__, static_url_path='/static', static_folder='static')

# ---------------------------------------------------------------------------
#  GLOBAL STATE
# ---------------------------------------------------------------------------
stop_training = False
is_training = False
training_lock = threading.Lock()
current_results = {}

# ---------------------------------------------------------------------------
#  BUILT-IN DATASETS
# ---------------------------------------------------------------------------
DATASETS = {}

def load_datasets():
    """Load all built-in datasets (lazy, called on first request)."""
    global DATASETS
    if DATASETS:
        return

    from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits

    # ── Image datasets ──────────────────────────────────────────────────

    (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
    DATASETS['mnist'] = {
        'x_train': x_tr / 255.0,
        'y_train': y_tr,
        'x_test': x_te / 255.0,
        'y_test': y_te,
        'input_shape': (28, 28),
        'num_classes': 10,
        'labels': [str(i) for i in range(10)],
        'name': 'MNIST Digits',
        'channels': 1,
        'type': 'image',
        'description': '70,000 handwritten digits (28x28 grayscale). The classic ML benchmark.'
    }

    (x_tr, y_tr), (x_te, y_te) = fashion_mnist.load_data()
    DATASETS['fashion_mnist'] = {
        'x_train': x_tr / 255.0,
        'y_train': y_tr,
        'x_test': x_te / 255.0,
        'y_test': y_te,
        'input_shape': (28, 28),
        'num_classes': 10,
        'labels': ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
        'name': 'Fashion-MNIST',
        'channels': 1,
        'type': 'image',
        'description': '70,000 clothing images across 10 categories. Harder than MNIST.'
    }

    (x_tr, y_tr), (x_te, y_te) = cifar10.load_data()
    y_tr, y_te = y_tr.flatten(), y_te.flatten()
    DATASETS['cifar10'] = {
        'x_train': x_tr / 255.0,
        'y_train': y_tr,
        'x_test': x_te / 255.0,
        'y_test': y_te,
        'input_shape': (32, 32, 3),
        'num_classes': 10,
        'labels': ['Airplane', 'Auto', 'Bird', 'Cat', 'Deer',
                   'Dog', 'Frog', 'Horse', 'Ship', 'Truck'],
        'name': 'CIFAR-10',
        'channels': 3,
        'type': 'image',
        'description': '60,000 color photos (32x32) across 10 categories. Needs CNN to do well.'
    }

    # ── sklearn 8x8 digit images ─────────────────────────────────────────
    dig = load_digits()
    X_dig = dig.data / 16.0
    y_dig = dig.target
    split = int(len(X_dig) * 0.8)
    idx = np.random.RandomState(42).permutation(len(X_dig))
    X_dig, y_dig = X_dig[idx], y_dig[idx]
    DATASETS['digits'] = {
        'x_train': X_dig[:split].reshape(-1, 8, 8),
        'y_train': y_dig[:split],
        'x_test': X_dig[split:].reshape(-1, 8, 8),
        'y_test': y_dig[split:],
        'input_shape': (8, 8),
        'num_classes': 10,
        'labels': [str(i) for i in range(10)],
        'name': 'Digits (8x8)',
        'channels': 1,
        'type': 'image',
        'description': '1,797 tiny digit images (8x8 grayscale). Fast to train, good for quick experiments.'
    }

    # ── Tabular datasets ─────────────────────────────────────────────────

    def make_tabular(loader, name, desc, labels=None):
        ds = loader()
        X = ds.data.astype(np.float32)
        y = ds.target.astype(np.int32)
        # normalize
        mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-8
        X = (X - mu) / sigma
        # split
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]
        split = int(len(X) * 0.8)
        lbl = labels if labels else [str(i) for i in range(len(np.unique(y)))]
        return {
            'x_train': X[:split], 'y_train': y[:split],
            'x_test': X[split:], 'y_test': y[split:],
            'input_shape': (X.shape[1],),
            'num_classes': len(np.unique(y)),
            'labels': lbl,
            'name': name,
            'channels': 0,
            'type': 'tabular',
            'description': desc,
            'feature_names': list(ds.feature_names) if hasattr(ds, 'feature_names') else [f'f{i}' for i in range(X.shape[1])]
        }

    DATASETS['iris'] = make_tabular(
        load_iris, 'Iris Flowers',
        '150 samples, 4 features (sepal/petal length+width), 3 species. The original ML dataset.',
        ['Setosa', 'Versicolor', 'Virginica']
    )

    DATASETS['wine'] = make_tabular(
        load_wine, 'Wine Quality',
        '178 wines, 13 chemical features (alcohol, malic acid, etc.), 3 quality classes.',
        ['Class 0', 'Class 1', 'Class 2']
    )

    DATASETS['breast_cancer'] = make_tabular(
        load_breast_cancer, 'Breast Cancer',
        '569 tumors, 30 features (radius, texture, etc.), 2 classes (malignant/benign).',
        ['Malignant', 'Benign']
    )


# ---------------------------------------------------------------------------
#  MODEL BUILDER
# ---------------------------------------------------------------------------
def get_regularizer(reg_type, reg_rate):
    if reg_type == 'l1':
        return tf.keras.regularizers.l1(reg_rate)
    elif reg_type == 'l2':
        return tf.keras.regularizers.l2(reg_rate)
    return None


def get_optimizer(name, lr):
    opts = {
        'adam': Adam,
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adagrad': Adagrad
    }
    return opts.get(name, Adam)(learning_rate=lr)


def build_model(input_shape, num_classes, layers_config, activation,
                regularization, reg_rate, dropout, use_cnn=False):
    """Build a model from the layer configuration."""
    model = Sequential()
    reg = get_regularizer(regularization, reg_rate)

    if use_cnn and len(input_shape) >= 2:
        if len(input_shape) == 2:
            model.add(Reshape((*input_shape, 1), input_shape=input_shape))
        else:
            model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

        model.add(Conv2D(32, (3, 3), activation=activation, padding='same',
                         kernel_regularizer=reg))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation=activation, padding='same',
                         kernel_regularizer=reg))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
    else:
        model.add(Flatten(input_shape=input_shape))

    for neurons in layers_config:
        model.add(Dense(neurons, activation=activation, kernel_regularizer=reg))
        if dropout > 0:
            model.add(Dropout(dropout))

    model.add(Dense(num_classes, activation='softmax'))
    return model


# ---------------------------------------------------------------------------
#  ROUTES
# ---------------------------------------------------------------------------
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')


@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """Return available datasets with metadata."""
    load_datasets()
    info = {}
    for key, ds in DATASETS.items():
        info[key] = {
            'name': ds['name'],
            'input_shape': list(ds['input_shape']),
            'num_classes': ds['num_classes'],
            'labels': ds['labels'],
            'train_size': len(ds['x_train']),
            'test_size': len(ds['x_test']),
            'channels': ds['channels'],
            'type': ds.get('type', 'image'),
            'description': ds.get('description', ''),
            'feature_names': ds.get('feature_names', [])
        }
    return jsonify(info)


@app.route('/api/dataset/samples', methods=['POST'])
def dataset_samples():
    """Return sample images/data from a dataset for preview."""
    load_datasets()
    data = request.json
    ds_key = data.get('dataset', 'mnist')

    if ds_key not in DATASETS:
        return jsonify({'error': 'Unknown dataset'}), 400

    ds = DATASETS[ds_key]
    n_samples = min(16, len(ds['x_test']))
    indices = np.random.choice(len(ds['x_test']), size=n_samples, replace=False)
    samples = []
    for idx in indices:
        img = ds['x_test'][idx]
        label = int(ds['y_test'][idx])
        samples.append({
            'pixels': img.tolist(),
            'label': label,
            'label_name': ds['labels'][label]
        })
    return jsonify({
        'samples': samples,
        'shape': list(ds['input_shape']),
        'channels': ds['channels'],
        'type': ds.get('type', 'image')
    })


@app.route('/api/dataset/visualize', methods=['POST'])
def dataset_visualize():
    """Return dataset statistics and visualization data."""
    load_datasets()
    data = request.json
    ds_key = data.get('dataset', 'mnist')

    if ds_key not in DATASETS:
        return jsonify({'error': 'Unknown dataset'}), 400

    ds = DATASETS[ds_key]
    x_all = np.concatenate([ds['x_train'], ds['x_test']])
    y_all = np.concatenate([ds['y_train'], ds['y_test']])

    # class distribution
    unique, counts = np.unique(y_all, return_counts=True)
    class_dist = [{'label': ds['labels'][int(u)], 'count': int(c)}
                  for u, c in zip(unique, counts)]

    result = {
        'class_distribution': class_dist,
        'total_samples': len(x_all),
        'train_size': len(ds['x_train']),
        'test_size': len(ds['x_test']),
        'input_shape': list(ds['input_shape']),
        'num_classes': ds['num_classes'],
        'type': ds.get('type', 'image'),
        'description': ds.get('description', ''),
        'feature_names': ds.get('feature_names', [])
    }

    # for tabular data: feature statistics + PCA 2D projection
    if ds.get('type') == 'tabular':
        x_flat = x_all.reshape(len(x_all), -1)
        feat_names = ds.get('feature_names', [f'f{i}' for i in range(x_flat.shape[1])])

        # feature stats
        stats = []
        for i in range(min(x_flat.shape[1], 30)):
            col = x_flat[:, i]
            stats.append({
                'name': feat_names[i] if i < len(feat_names) else f'f{i}',
                'mean': round(float(col.mean()), 3),
                'std': round(float(col.std()), 3),
                'min': round(float(col.min()), 3),
                'max': round(float(col.max()), 3)
            })
        result['feature_stats'] = stats

        # PCA 2D scatter for visualization
        if x_flat.shape[1] >= 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(x_flat)
            # subsample for performance
            n_vis = min(500, len(coords))
            vis_idx = np.random.choice(len(coords), n_vis, replace=False)
            scatter = []
            for i in vis_idx:
                scatter.append({
                    'x': round(float(coords[i, 0]), 4),
                    'y': round(float(coords[i, 1]), 4),
                    'label': int(y_all[i]),
                    'label_name': ds['labels'][int(y_all[i])]
                })
            result['pca_scatter'] = scatter
            result['pca_variance'] = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    # for image data: pixel mean heatmap
    elif ds.get('type') == 'image':
        # per-class mean images
        class_means = []
        for c in range(ds['num_classes']):
            mask = y_all == c
            if mask.sum() > 0:
                mean_img = x_all[mask].mean(axis=0)
                class_means.append({
                    'label': ds['labels'][c],
                    'pixels': mean_img.tolist()
                })
        result['class_means'] = class_means

    return jsonify(result)


@app.route('/api/upload', methods=['POST'])
def upload_dataset():
    """Accept a CSV file as a custom dataset. Last column = labels."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    content = file.read().decode('utf-8')
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)

    if len(rows) < 10:
        return jsonify({'error': 'Need at least 10 rows'}), 400

    # detect header
    try:
        float(rows[0][0])
        header = None
        feature_names = [f'feature_{i}' for i in range(len(rows[0]) - 1)]
    except ValueError:
        header = rows[0]
        feature_names = header[:-1]
        rows = rows[1:]

    data_arr = np.array(rows, dtype=float)
    X = data_arr[:, :-1].astype(np.float32)
    y = data_arr[:, -1].astype(np.int32)

    # normalize
    mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-8
    X = (X - mu) / sigma

    num_classes = len(np.unique(y))
    input_shape = (X.shape[1],)

    split = int(len(X) * 0.8)
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]

    DATASETS['custom'] = {
        'x_train': X[:split],
        'y_train': y[:split],
        'x_test': X[split:],
        'y_test': y[split:],
        'input_shape': input_shape,
        'num_classes': num_classes,
        'labels': [str(i) for i in range(num_classes)],
        'name': f'Custom ({file.filename})',
        'channels': 0,
        'type': 'tabular',
        'description': f'Uploaded file: {file.filename}. {len(X)} samples, {X.shape[1]} features, {num_classes} classes.',
        'feature_names': feature_names
    }

    return jsonify({
        'status': 'uploaded',
        'name': file.filename,
        'samples': len(X),
        'features': X.shape[1],
        'classes': num_classes,
        'labels': [str(i) for i in range(num_classes)]
    })


@app.route('/api/train', methods=['POST'])
def train_model():
    """Train model and stream epoch results via SSE."""
    global stop_training, is_training, current_results

    if is_training:
        return jsonify({'error': 'Training already in progress'}), 409

    load_datasets()
    data = request.json

    ds_key = data.get('dataset', 'mnist')
    if ds_key not in DATASETS:
        return jsonify({'error': 'Unknown dataset'}), 400

    ds = DATASETS[ds_key]

    learning_rate = float(data.get('learning_rate', 0.001))
    activation = data.get('activation', 'relu')
    regularization = data.get('regularization', 'none')
    reg_rate = float(data.get('reg_rate', 0.01))
    optimizer_name = data.get('optimizer', 'adam')
    batch_size = int(data.get('batch_size', 32))
    epochs = int(data.get('epochs', 20))
    train_split = float(data.get('train_split', 0.8))
    dropout = float(data.get('dropout', 0.0))
    layers_config = data.get('layers', [128, 64])
    use_cnn = data.get('use_cnn', False)

    split_idx = int(len(ds['x_train']) * train_split)
    x_tr = ds['x_train'][:split_idx]
    y_tr = ds['y_train'][:split_idx]
    x_val = ds['x_train'][split_idx:]
    y_val = ds['y_train'][split_idx:]
    x_test = ds['x_test']
    y_test = ds['y_test']

    model = build_model(
        ds['input_shape'], ds['num_classes'], layers_config,
        activation, regularization, reg_rate, dropout, use_cnn
    )
    model.compile(
        optimizer=get_optimizer(optimizer_name, learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    total_params = model.count_params()

    stop_training = False
    is_training = True
    current_results = {
        'accuracy': [], 'val_accuracy': [],
        'loss': [], 'val_loss': []
    }

    def generate():
        global stop_training, is_training, current_results

        yield f"data: {json.dumps({'type': 'model_info', 'summary': summary_lines, 'total_params': total_params, 'total_epochs': epochs})}\n\n"

        try:
            best_val_acc = 0
            for epoch in range(1, epochs + 1):
                if stop_training:
                    yield f"data: {json.dumps({'type': 'stopped', 'epoch': epoch - 1})}\n\n"
                    break

                t0 = time.time()
                hist = model.fit(
                    x_tr, y_tr,
                    batch_size=batch_size,
                    epochs=1,
                    validation_data=(x_val, y_val),
                    verbose=0
                )
                elapsed = round(time.time() - t0, 2)

                acc = round(float(hist.history['accuracy'][0]), 5)
                val_acc = round(float(hist.history['val_accuracy'][0]), 5)
                loss = round(float(hist.history['loss'][0]), 5)
                val_loss = round(float(hist.history['val_loss'][0]), 5)

                current_results['accuracy'].append(acc)
                current_results['val_accuracy'].append(val_acc)
                current_results['loss'].append(loss)
                current_results['val_loss'].append(val_loss)

                best_val_acc = max(best_val_acc, val_acc)

                epoch_data = {
                    'type': 'epoch',
                    'epoch': epoch,
                    'accuracy': acc,
                    'val_accuracy': val_acc,
                    'loss': loss,
                    'val_loss': val_loss,
                    'best_val_accuracy': round(best_val_acc, 5),
                    'elapsed': elapsed
                }
                yield f"data: {json.dumps(epoch_data)}\n\n"

            # final test evaluation
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            y_pred = model.predict(x_test, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)

            cm = confusion_matrix(y_test, y_pred_classes).tolist()

            # sample predictions
            sample_indices = np.random.choice(len(x_test), size=min(16, len(x_test)), replace=False)
            samples = []
            for idx in sample_indices:
                img = x_test[idx]
                true_label = int(y_test[idx])
                pred_label = int(y_pred_classes[idx])
                confidence = round(float(y_pred[idx][pred_label]) * 100, 1)
                samples.append({
                    'pixels': img.tolist(),
                    'true_label': true_label,
                    'true_name': ds['labels'][true_label],
                    'pred_label': pred_label,
                    'pred_name': ds['labels'][pred_label],
                    'confidence': confidence,
                    'correct': true_label == pred_label
                })

            # per-class accuracy
            per_class = []
            for c in range(ds['num_classes']):
                mask = y_test == c
                if mask.sum() > 0:
                    class_acc = round(float((y_pred_classes[mask] == c).mean()) * 100, 1)
                else:
                    class_acc = 0
                per_class.append({
                    'label': ds['labels'][c],
                    'accuracy': class_acc,
                    'count': int(mask.sum())
                })

            final_data = {
                'type': 'complete',
                'test_accuracy': round(float(test_acc), 5),
                'test_loss': round(float(test_loss), 5),
                'confusion_matrix': cm,
                'labels': ds['labels'],
                'samples': samples,
                'per_class': per_class,
                'shape': list(ds['input_shape']),
                'channels': ds.get('channels', 1)
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            is_training = False

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/stop', methods=['POST'])
def stop_training_route():
    global stop_training
    stop_training = True
    return jsonify({'status': 'Training stop requested'})


if __name__ == '__main__':
    app.run(debug=False, threaded=True, port=5000)
