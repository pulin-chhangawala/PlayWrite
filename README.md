<p align="center">
  <h1 align="center">🧠 playwrite</h1>
  <p align="center"><b>A neural network playground you run locally.</b></p>
  <p align="center">
    Pick a dataset. Build an architecture. Tune the knobs. Hit train.<br>
    Watch accuracy climb in real time, then dig into where your model gets it right and wrong.
  </p>
</p>

---

## What It Does

Playwrite is an interactive web app for building, training, and evaluating neural networks without writing any code. It runs a Flask server on your machine and opens a browser UI where you can:

1. **Choose a dataset** from 7 built-in options (or upload your own CSV)
2. **Explore the data** with class distributions, PCA projections, and per-class mean images
3. **Design your model** by stacking dense layers and toggling CNN feature extraction
4. **Train with live feedback** as accuracy and loss curves update epoch by epoch
5. **Analyze results** through confusion matrices, per-class breakdowns, and sample predictions with confidence scores

No notebooks required. No command-line flags. Just `pip install` and go.

---

## Getting Started

```bash
pip install -r requirements.txt
python Backend.py
```

Then open **http://localhost:5000** in your browser.

### Requirements

- Python 3.8+
- TensorFlow
- Flask
- NumPy
- scikit-learn

---

## Datasets

### Image

| Name | Samples | Dimensions | Classes | What It Is |
|------|---------|------------|---------|------------|
| MNIST | 70,000 | 28 x 28 | 10 | Handwritten digits (0 through 9) |
| Fashion-MNIST | 70,000 | 28 x 28 | 10 | Clothing items (T-shirts, sneakers, bags, etc.) |
| CIFAR-10 | 60,000 | 32 x 32 x 3 | 10 | Color photos (planes, cars, birds, cats, etc.) |
| Digits (8x8) | 1,797 | 8 x 8 | 10 | Tiny digit thumbnails from scikit-learn |

### Tabular

| Name | Samples | Features | Classes | What It Is |
|------|---------|----------|---------|------------|
| Iris | 150 | 4 | 3 | Petal and sepal measurements for 3 flower species |
| Wine | 178 | 13 | 3 | Chemical composition of wines from 3 different cultivars |
| Breast Cancer | 569 | 30 | 2 | Tumor cell measurements, malignant vs. benign |

### Custom Upload

Upload any CSV file where the **last column contains the labels**. The app normalizes features, shuffles, and splits the data automatically.

---

## What You Can Configure

**Architecture**
- Stack any number of dense (fully connected) layers with 8 to 1,024 neurons each
- Toggle CNN mode to prepend two Conv2D + MaxPooling layers before the dense stack (useful for image data)
- Set dropout between 0.0 and 0.5

**Training**
- Optimizer: Adam, SGD, RMSprop, or Adagrad
- Activation function: ReLU, Sigmoid, Tanh, ELU, or SELU
- Regularization: None, L1, or L2 (with adjustable strength)
- Batch size: 16, 32, 64, 128, or 256
- Epochs: 1 to 1,000
- Train/validation split ratio: 10% to 95%

---

## Visualizations

### Before Training

When you select a dataset, the **Dataset Explorer** panel shows:

- **Class distribution chart** with sample counts per category
- **Per-class mean images** (image datasets) showing the average pixel pattern for each label
- **PCA scatter plot** (tabular datasets) projecting samples into 2D to reveal how classes cluster
- **Feature statistics table** (tabular datasets) with mean, standard deviation, min, and max for each variable

### During Training

Results stream to the browser in real time (via fetch streaming):

- Train and validation **accuracy curves**, updated every epoch
- Train and validation **loss curves**
- Progress bar with epoch counter and elapsed time
- Live stats: current accuracy, current loss, best validation accuracy so far

You can **stop training at any point** without losing the results gathered so far.

### After Training

Once the final epoch completes, the app evaluates the model on the held-out test set and shows:

- **Confusion matrix** heatmap (diagonal = correct predictions, off-diagonal = errors)
- **Per-class accuracy** horizontal bars so you can spot which categories the model struggles with
- **Sample predictions** grid showing actual data points alongside predicted labels, confidence scores, and whether each prediction was correct
- **Model summary** with the full layer-by-layer architecture and total parameter count

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/api/datasets` | List all available datasets with metadata |
| `POST` | `/api/dataset/samples` | Get sample data points for preview |
| `POST` | `/api/dataset/visualize` | Get class distribution, PCA projection, feature stats, mean images |
| `POST` | `/api/upload` | Upload a CSV as a custom dataset |
| `POST` | `/api/train` | Start training (response is a stream of epoch results) |
| `POST` | `/api/stop` | Stop training early |

---

## Project Layout

```
playwrite/
├── Backend.py            Server, model builder, training loop, dataset loading
├── index.html            Page layout with sidebar controls and visualization panels
├── requirements.txt      Python dependencies
└── static/
    ├── style.css         Dark theme with glassmorphism, responsive grid
    └── app.js            Frontend logic, Chart.js charts, canvas rendering
```

## Built With

**Backend**: Flask, TensorFlow/Keras, scikit-learn, NumPy
**Frontend**: Vanilla HTML/CSS/JS, Chart.js 4, Google Fonts (Inter, JetBrains Mono)

---

## License

GNU General Public License v3.0
