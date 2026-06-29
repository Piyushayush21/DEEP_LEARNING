
"""
Artificial Neural Network (ANN) Classification
Dataset: Iris (3 classes: Setosa, Versicolor, Virginica)
Framework: TensorFlow / Keras
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------
# 1. Load and explore the data
# ----------------------------------------------------------------------
iris = load_iris()
X = iris.data                      # shape: (150, 4)
y = iris.target                    # shape: (150,) -> values 0, 1, 2
feature_names = iris.feature_names
class_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["species"] = pd.Categorical.from_codes(y, class_names)

print("Dataset shape:", X.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nClass distribution:\n", df["species"].value_counts())

# ----------------------------------------------------------------------
# 2. Preprocess: split, scale, encode
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-hot encode labels for multi-class classification with softmax output
num_classes = len(class_names)
y_train_oh = keras.utils.to_categorical(y_train, num_classes)
y_test_oh = keras.utils.to_categorical(y_test, num_classes)

print(f"\nTrain size: {X_train_scaled.shape[0]}, Test size: {X_test_scaled.shape[0]}")

# ----------------------------------------------------------------------
# 3. Build the ANN
# ----------------------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ----------------------------------------------------------------------
# 4. Train
# ----------------------------------------------------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True
)

history = model.fit(
    X_train_scaled, y_train_oh,
    validation_split=0.15,
    epochs=200,
    batch_size=8,
    callbacks=[early_stop],
    verbose=0,
)

print(f"\nTraining stopped after {len(history.history['loss'])} epochs.")

# ----------------------------------------------------------------------
# 5. Evaluate
# ----------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test_scaled, y_test_oh, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

y_pred_probs = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# ----------------------------------------------------------------------
# 6. Plots: training curves + confusion matrix
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].plot(history.history["loss"], label="train loss")
axes[0].plot(history.history["val_loss"], label="val loss")
axes[0].set_title("Loss over epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="train acc")
axes[1].plot(history.history["val_accuracy"], label="val acc")
axes[1].set_title("Accuracy over epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=axes[2], colorbar=False, cmap="Blues")
axes[2].set_title("Confusion Matrix (Test Set)")

plt.tight_layout()
plt.savefig("/home/claude/ann_results.png", dpi=150)
print("\nSaved plots to ann_results.png")

# ----------------------------------------------------------------------
# 7. Example single prediction
# ----------------------------------------------------------------------
sample = X_test_scaled[0:1]
pred_probs = model.predict(sample, verbose=0)[0]
pred_class = class_names[np.argmax(pred_probs)]
true_class = class_names[y_test[0]]

print(f"\nExample prediction -> Predicted: {pred_class}, Actual: {true_class}")
print("Class probabilities:", dict(zip(class_names, np.round(pred_probs, 3))))
