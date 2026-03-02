# playwrite

**Interactive neural network playground.** Build, train, and visualize neural networks right in your browser. Comes with 7 built-in datasets, a dynamic architecture builder, real-time training curves, and rich post-training analysis.

## Quick Start

```bash
pip install -r requirements.txt
python Backend.py
# open http://localhost:5000
```

## Built-In Datasets

### Image Datasets
| Dataset | Samples | Shape | Classes | Description |
|---------|---------|-------|---------|-------------|
| **MNIST** | 70,000 | 28x28 grayscale | 10 | Handwritten digits, the classic ML benchmark |
| **Fashion-MNIST** | 70,000 | 28x28 grayscale | 10 | Clothing items (T-shirts, shoes, bags, etc.) |
| **CIFAR-10** | 60,000 | 32x32 RGB | 10 | Color photos (planes, cars, birds, cats, etc.) |
| **Digits (8x8)** | 1,797 | 8x8 grayscale | 10 | Tiny digit images, great for quick experiments |

### Tabular Datasets
| Dataset | Samples | Features | Classes | Description |
|---------|---------|----------|---------|-------------|
| **Iris** | 150 | 4 | 3 | Sepal/petal measurements for 3 flower species |
| **Wine** | 178 | 13 | 3 | Chemical analysis of wines from 3 cultivars |
| **Breast Cancer** | 569 | 30 | 2 | Tumor features for malignant/benign classification |

You can also **upload your own CSV** file. The last column is treated as labels.

## Features

### Dataset Explorer
When you select a dataset, the main panel shows:
- **Class distribution** bar chart showing sample counts per category
- **Per-class mean images** (image datasets) showing average pixel intensity for each class
- **PCA 2D scatter plot** (tabular datasets) showing how well classes separate in reduced dimensions
- **Feature statistics table** (tabular datasets) with mean, std, min, max for each feature
- Dataset info badges: total samples, input dimensions, number of classes, data type

### Configurable Architecture
- Add and remove dense layers with adjustable neuron counts
- Optional CNN feature extraction (Conv2D + MaxPooling) for image datasets
- Dropout regularization via slider (0 to 0.5)
- 5 activation functions: ReLU, Sigmoid, Tanh, ELU, SELU

### Hyperparameter Control
- **Optimizers**: Adam, SGD, RMSprop, Adagrad
- **Regularization**: None, L1, L2 with adjustable rate
- **Batch size**: 16 to 256
- **Epochs**: 1 to 200
- **Train/validation split**: adjustable ratio (10% to 95%)

### Real-Time Training
Training streams results to the browser epoch-by-epoch via Server-Sent Events:
- **Live accuracy curve** (train vs. validation)
- **Live loss curve** (train vs. validation)
- **Progress bar** with elapsed time and epoch counter
- **Stats dashboard**: current train accuracy, validation accuracy, loss, and best validation accuracy
- **Stop button** to halt training mid-run

### Post-Training Analysis
After training completes:
- **Confusion matrix** heatmap showing where the model confuses classes (diagonal = correct, off-diagonal = errors)
- **Per-class accuracy bars** revealing which categories the model handles best and worst
- **Sample predictions grid** with actual images/data, predicted labels, confidence percentages, and correct/incorrect indicators
- **Model summary** with layer-by-layer architecture and total parameter count

## API

```
GET  /api/datasets          List available datasets with metadata
POST /api/dataset/samples   Get sample data for preview
POST /api/dataset/visualize Get class distribution, PCA, feature stats, mean images
POST /api/upload            Upload a custom CSV dataset
POST /api/train             Train model (streams epoch results via SSE)
POST /api/stop              Stop training
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask (Python) |
| ML Framework | TensorFlow / Keras |
| Data Science | scikit-learn (PCA, confusion matrix, datasets) |
| Frontend | Vanilla HTML + CSS + JS |
| Charts | Chart.js 4 |
| Fonts | Inter + JetBrains Mono (Google Fonts) |

## Project Structure

```
playwrite/
  Backend.py          600 lines   Flask server, model builder, SSE, dataset management
  index.html          244 lines   Two-panel layout, sidebar config, visualization panels
  static/
    style.css         853 lines   Dark glassmorphism theme, responsive grid, chart styles
    app.js            672 lines   SSE handler, Chart.js, layer builder, canvas renderer
  requirements.txt      4 lines   Python dependencies
```

**Total: 2,373 lines**

## License

GNU General Public License v3.0
