# playwrite

**Interactive neural network playground.** Train models on MNIST, Fashion-MNIST, CIFAR-10, or your own data. Watch training happen in real time with live accuracy and loss curves, then explore confusion matrices, per-class accuracy breakdowns, and sample predictions.

## Quick Start

```bash
pip install -r requirements.txt
python Backend.py
# open http://localhost:5000
```

## Features

### Multiple Built-In Datasets
- **MNIST**: 70,000 handwritten digits (28x28 grayscale)
- **Fashion-MNIST**: 70,000 clothing items across 10 categories
- **CIFAR-10**: 60,000 color images across 10 categories (32x32 RGB)
- **Custom CSV Upload**: bring your own tabular data (last column = labels)

### Configurable Architecture
- Add and remove dense layers with adjustable neuron counts
- Optional CNN feature extraction (Conv2D + MaxPooling) for image data
- Dropout regularization with visual slider
- 5 activation functions: ReLU, Sigmoid, Tanh, ELU, SELU

### Hyperparameter Control
- **Optimizers**: Adam, SGD, RMSprop, Adagrad
- **Regularization**: None, L1, L2 with adjustable rate
- **Batch size**: 16 to 256
- **Epochs**: 1 to 200
- **Train/validation split**: adjustable ratio

### Real-Time Training Visualization
Training results stream to the browser epoch-by-epoch via Server-Sent Events:
- **Live accuracy curve**: train vs. validation accuracy, updated every epoch
- **Live loss curve**: train vs. validation loss
- **Progress bar** with elapsed time
- **Stats dashboard**: current accuracy, loss, and best validation accuracy

### Post-Training Analysis
After training completes, the app automatically generates:
- **Confusion matrix heatmap**: see where the model confuses classes
- **Per-class accuracy bars**: which categories does the model handle best/worst?
- **Sample predictions grid**: actual images with predicted labels, confidence scores, and correct/incorrect indicators
- **Model summary**: full layer-by-layer architecture with parameter counts

## Architecture

```
Frontend (index.html + app.js + style.css)
    |
    | SSE stream (text/event-stream)
    |
Backend (Flask + TensorFlow)
    |
    +-- /api/datasets      GET   list available datasets
    +-- /api/dataset/samples POST  get sample images for preview
    +-- /api/upload         POST  upload custom CSV dataset
    +-- /api/train          POST  train model (streams epoch results)
    +-- /api/stop           POST  stop training
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask (Python) |
| ML Framework | TensorFlow / Keras |
| Frontend | Vanilla HTML + CSS + JS |
| Charts | Chart.js 4 |
| Fonts | Inter + JetBrains Mono |

## Project Structure

| File | Lines | Purpose |
|------|-------|---------|
| `Backend.py` | 437 | Flask server, model builder, SSE training loop, dataset management |
| `index.html` | 236 | Two-panel layout, configuration sidebar, visualization panels |
| `static/style.css` | 760 | Dark glassmorphism theme, responsive grid, chart styling |
| `static/app.js` | 671 | SSE handler, Chart.js integration, layer builder, image renderer |
| `requirements.txt` | 4 | Python dependencies |

## License

GNU General Public License v3.0
