# MNIST Model Trainer

## Overview

The MNIST Model Trainer is a web application that allows users to train a neural network on the MNIST dataset using TensorFlow and Flask. The application provides an intuitive interface for adjusting hyperparameters and visualizing training progress, making it a useful tool for both educational and practical purposes in machine learning.

## Features

- **User-Friendly Interface**: Easy-to-use web interface for specifying hyperparameters.
- **Dynamic Training Configuration**: Users can set learning rate, activation function, regularization type, regularization rate, and training data split percentage.
- **Training Visualization**: Updates on training and validation accuracy displayed in a line chart and displays the latest values as numeric figures.
- **Stop Training Functionality**: Ability to halt training midway.
  
## Technologies Used

- **Backend**: Flask (Python)
- **Machine Learning Framework**: TensorFlow
- **Frontend**: HTML, CSS, JavaScript
- **Charting Library**: Chart.js

## Installation

### Prerequisites

- Python 3.x
- All neccessary python extensions as mentioned in the backend.py file
- (optional) frontend development interface

### Clone the Repository

```bash
git clone https://github.com/yourusername/mnist-model-trainer.git
cd mnist-model-trainer
```

### Set Up a Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies

```bash
pip install Flask tensorflow
```

### Run the Application

1. **Start the Flask Server**:

```bash
python Backend.py
```

2. **Open the Web Interface**: Open your web browser and navigate to `http://127.0.0.1:5000`.

## Usage

1. **Input Hyperparameters**: Fill in the form with desired hyperparameters:
   - Learning Rate: Set a value (e.g., 0.001).
   - Activation Function: Choose from ReLU, Sigmoid, or Tanh.
   - Regularization: Select L1 or L2.
   - Regularization Rate: Set a value (e.g., 0.01).
   - Training Data Split: Specify the percentage of data to be used for training.

2. **Train the Model**: Click the "Train Model" button to start the training process.

3. **Stop Training**: Wait for atleast 30 seconds, and click the "Stop Training" button to halt the training.

4. **View Results**: After training, the latest accuracy and validation accuracy will be displayed. A graph will show the accuracy over epochs.

## API Endpoints

### `/train` (POST)

Trains the model with specified hyperparameters.

**Request Body**:
```json
{
  "learning_rate": float,
  "activation": "relu" | "sigmoid" | "tanh",
  "regularization": "l1" | "l2",
  "reg_rate": float,
  "train_split": float
}
```

**Response**:
```json
{
  "latest_accuracy": float,
  "latest_val_accuracy": float,
  "accuracy": [float],
  "val_accuracy": [float]
}
```

### `/stop` (POST)

Stops the training process.

**Response**:
```json
{
  "status": "Training stopped"
}
```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/YourFeature`).
6. Open a pull request.

## License

This project is licensed under the GNU General Public License v3.0. See the LICENSE.md file for details.

## Acknowledgments

- Thanks to the developers of TensorFlow and Flask for creating powerful frameworks for machine learning and web development.
- Inspired by various online resources and tutorials on machine learning and web application development.

## Contact

For any questions or suggestions, please reach out to [pulinchhangawala@gmail.com].
