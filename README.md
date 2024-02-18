# Time Series Machine Learning Model with TensorFlow

This project focuses on building a machine learning model using TensorFlow for time series forecasting. The model is trained on a dataset containing daily tomato prices. The goal is to predict the future tomato prices based on historical data.

## Features

- Utilizes LSTM (Long Short-Term Memory) neural network architecture for time series prediction.
- Implements windowed datasets for training and validation.
- Visualizes model performance using training and validation loss curves.
- Provides a custom callback to stop training when desired performance is achieved.

## Dataset

The dataset used in this project can be downloaded from the following link:

[Download Dataset](https://www.kaggle.com/datasets/ramkrijal/tomato-daily-prices/download?datasetVersionNumber=2)

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib

## How to Run

1. Download the dataset from the provided link and save it to the project directory.
2. Install all necessary dependencies by running `pip install -r requirements.txt`.
3. Run the Python script in an environment that supports TensorFlow.
4. Follow the instructions in the script to preprocess the data, train the model, and evaluate its performance.

## Usage

- Train your own time series forecasting model using the provided script.
- Visualize the training and validation loss curves to monitor model performance.
- Utilize the trained model to make predictions on future tomato prices.
