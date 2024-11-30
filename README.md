# Drone Sensor Anomaly Detection Using Transformers

This project implements a Transformer-based model to classify anomalies in drone sensor data logs. The model leverages learnable positional encodings to handle sequential data efficiently.

## Features
- **Sensor Data Preprocessing**: Standardizes sensor values for model input.
- **Transformer Model**: Utilizes a Transformer with learnable positional encoding for anomaly classification.
- **Streamlit Interface**: Provides an interactive web app for training and predicting anomalies.

## Requirements
Ensure the following dependencies are installed:
- Python 3.8+
- `torch`
- `scikit-learn`
- `pandas`
- `numpy`
- `streamlit`

Install required packages:
```bash
pip install torch scikit-learn pandas numpy streamlit
```
How to Use
## 1. Run the App
Start the Streamlit app:
```bash
streamlit run app.py
```
## 2. Train the Model
Navigate to the "Train Model" menu.
Upload a CSV file containing the drone sensor data.
Ensure the dataset includes the following features:
tx, rx, txspeed, rxspeed, cpu, latitude, longitude, altitude, x gyro, y gyro, z gyro, and a label column.
Adjust the number of training epochs using the slider and click Train Model.
Once training is complete, the model's accuracy will be displayed.

## 3. Predict Anomalies
Navigate to the "Predict" menu.
Enter sensor values manually for prediction.
Click Predict to classify the input as "Normal" or "Anomaly."

## Dataset Format
The dataset must be a CSV file with the following columns:

Features: tx, rx, txspeed, rxspeed, cpu, latitude, longitude, altitude, x gyro, y gyro, z gyro
Label: label (binary: 0 for normal, 1 for anomaly)

tx,rx,txspeed,rxspeed,cpu,latitude,longitude,altitude,x gyro,y gyro,z gyro,label
100,200,50,50,10,40.7128,-74.0060,1000,0.1,0.2,0.3,0
150,250,60,55,20,40.7129,-74.0065,1200,0.2,0.3,0.4,1
...


## Model Details
Input: 11 drone sensor features.
Architecture:
Input embedding layer with d_model=64
Learnable positional encoding
Transformer encoder with 2 layers and 4 attention heads
Fully connected layer for classification


## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to the team for contributions and support.
