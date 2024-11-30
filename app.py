import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Define Transformer Model
#class TransformerClassifier(nn.Module):
    #def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2):
        #super(TransformerClassifier, self).__init__()
        #self.embedding = nn.Linear(input_dim, d_model)
        #self.transformer = nn.TransformerEncoder(
            #nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), num_layers=num_layers
        #)
        #self.fc = nn.Linear(d_model, num_classes)

    #def forward(self, x):
        #x = x.unsqueeze(1)  # Add sequence dimension (batch_size, 1, input_dim)
        #x = self.embedding(x)
        #x = self.transformer(x)
        #x = x.mean(dim=1)  # Reduce sequence dimension
        #return self.fc(x)
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2, max_seq_len=1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_len, d_model))  # Learnable positional encoding
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension (batch_size, seq_len=1, input_dim)
        x = self.embedding(x)  # Project to d_model dimensions
        x += self.positional_encoding[: x.size(1)]  # Add positional encoding (seq_len matches x)
        x = self.transformer(x)  # Pass through transformer encoder
        x = x.mean(dim=1)  # Pool over sequence dimension
        return self.fc(x)
# Initialize session state for model and scaler
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None

# Sidebar for menu selection
menu = st.sidebar.selectbox("Menu", ["Train Model", "Predict"])

# Train Model Menu
if menu == "Train Model":
    st.title("Train Transformer Model Untuk Deteksi Anomali Pada Drone")

    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", data.head())

        # Specify feature columns
        feature_cols = ['tx', 'rx', 'txspeed', 'rxspeed', 'cpu', 'latitude', 'longitude', 'altitude', 'x gyro', 'y gyro', 'z gyro']
        label_col = 'label'

        if set(feature_cols + [label_col]).issubset(data.columns):
            # Data preprocessing
            X = data[feature_cols].values
            y = data[label_col].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32)

            # Initialize model
            input_dim = len(feature_cols)
            num_classes = len(np.unique(y))
            model = TransformerClassifier(input_dim, num_classes)

            # Training
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            epochs = st.slider("Select number of epochs", 1, 50, 10)
            if st.button("Train Model"):
                for epoch in range(epochs):
                    model.train()
                    for X_batch, y_batch in train_loader:
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()

                st.success("Model training completed!")

                # Save model and scaler in session state
                st.session_state.model = model
                st.session_state.scaler = scaler

                # Evaluation
                model.eval()
                y_true = []
                y_pred = []
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        outputs = model(X_batch)
                        _, predicted = torch.max(outputs.data, 1)
                        y_true.extend(y_batch.tolist())
                        y_pred.extend(predicted.tolist())

                # Compute metrics
                precision = precision_score(y_true, y_pred, average="binary")
                recall = recall_score(y_true, y_pred, average="binary")
                f1 = f1_score(y_true, y_pred, average="binary")
                accuracy = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)

                st.write("**Metrics on Test Set:**")
                st.write(f"- **Accuracy**: {accuracy * 100:.2f}%")
                st.write(f"- **Precision**: {precision:.2f}")
                st.write(f"- **Recall**: {recall:.2f}")
                st.write(f"- **F1-Score**: {f1:.2f}")
        else:
            st.error("Dataset harus menyertakan fitur dan kolom label yang ditentukan.")

# Predict Menu
elif menu == "Predict":
    st.title("Prediksi Anomali Pada Drone Sensor Data")

    if st.session_state.model and st.session_state.scaler:
        st.write("Enter sensor values for prediction:")
        user_input = []
        feature_cols = ['tx', 'rx', 'txspeed', 'rxspeed', 'cpu', 'latitude', 'longitude', 'altitude', 'x gyro', 'y gyro', 'z gyro']

        for col in feature_cols:
            val = st.number_input(f"Masukkan Nilai Sensor Untuk {col}", value=0.0)
            user_input.append(val)

        if st.button("Predict"):
            scaler = st.session_state.scaler
            model = st.session_state.model
            user_input = scaler.transform([user_input])  # Scale the input
            user_input_tensor = torch.tensor(user_input, dtype=torch.float32)
            with torch.no_grad():
                output = model(user_input_tensor)
                _, prediction = torch.max(output.data, 1)
            st.write("Prediction:", "Malfungsi" if prediction.item() == 1 else "Normal")
    else:
        st.warning("Silahkan training model terlebih dahulu di menu 'Train Model'.")
