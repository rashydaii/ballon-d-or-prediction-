# ==========================================================
# Artificial Neural Network - Ballon d'Or Winner Prediction
# ==========================================================

# ===============================
# Import Libraries
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ===============================
# Load Dataset
# ===============================

df = pd.read_csv(r"C:\Users\Acer\Downloads\archive (3)\ballondor_performance_vs_pr.csv")      

print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nMissing Values")
print(df.isnull().sum())

# ===============================
# Handle Missing Values
# ===============================

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ===============================
# Encode Categorical Columns
# ===============================

encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# ===============================
# Create Target Variable
# ===============================

# Winner = Rank 1
df["Winner"] = (df["Rank"] == 1).astype(int)

print("\nWinner Distribution")
print(df["Winner"].value_counts())

# ===============================
# Features and Target
# ===============================

X = df.drop(columns=["Winner", "Rank"])

y = df["Winner"]

# ===============================
# Normalize Data
# ===============================

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

# ===============================
# Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# ===============================
# Build Neural Network
# ===============================

model = Sequential()

model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))

model.add(Dense(32, activation="relu"))

model.add(Dense(16, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

# ===============================
# Compile Model
# ===============================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# Early Stopping
# ===============================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# ===============================
# Train Model
# ===============================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# ===============================
# Evaluate Model
# ===============================

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy :", accuracy)

print("Test Loss :", loss)

# ===============================
# Prediction
# ===============================

y_prob = model.predict(X_test)

y_pred = (y_prob > 0.5).astype(int).flatten()

# ===============================
# Confusion Matrix
# ===============================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")

print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()

# ===============================
# Classification Report
# ===============================

print("\nClassification Report")

print(classification_report(y_test, y_pred))

print("Accuracy :", accuracy_score(y_test, y_pred))

# ===============================
# Accuracy Graph
# ===============================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training vs Validation Accuracy")

plt.legend()

plt.grid()

plt.show()

# ===============================
# Loss Graph
# ===============================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.grid()

plt.show()

# ===============================
# Hyperparameter Experiment 1
# ===============================

print("\n==============================")
print("Experiment 1")
print("==============================")

model1 = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(1, activation="sigmoid")
])

model1.compile(
    optimizer=Adam(0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model1.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    verbose=0
)

score1 = model1.evaluate(X_test, y_test, verbose=0)

print("Accuracy:", score1[1])

# ===============================
# Hyperparameter Experiment 2
# ===============================

print("\n==============================")
print("Experiment 2")
print("==============================")

model2 = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model2.compile(
    optimizer=Adam(0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model2.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    verbose=0
)

score2 = model2.evaluate(X_test, y_test, verbose=0)

print("Accuracy:", score2[1])

# ===============================
# Hyperparameter Experiment 3
# ===============================

print("\n==============================")
print("Experiment 3")
print("==============================")

model3 = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model3.compile(
    optimizer=Adam(0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model3.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    verbose=0
)

score3 = model3.evaluate(X_test, y_test, verbose=0)

print("Accuracy:", score3[1])

# ===============================
# Comparison
# ===============================

print("\n==============================")
print("Hyperparameter Comparison")
print("==============================")

print("Model 1 Accuracy :", score1[1])

print("Model 2 Accuracy :", score2[1])

print("Model 3 Accuracy :", score3[1])

print("\nBest Accuracy :", max(score1[1], score2[1], score3[1]))