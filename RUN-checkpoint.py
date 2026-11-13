# ============================================================
# MULTI-CLASS CANCER PREDICTION USING QKAN
# ============================================================

# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from sklearn.preprocessing import quantile_transform
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances

from imblearn.over_sampling import SMOTE
from collections import Counter

# Plot styling
plt.rcParams['font.family'] = 'Times new roman'
plt.rcParams['font.size'] = 16
plt.rcParams['font.weight'] = 'bold'

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("DATASET.csv")

istrain = 0        # 1 → train model, 0 → only test

# Feature matrix and label vector
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Save original gene names
gene_names = df.columns[1:-1]

print("Original Shape:", X.shape)

# ============================================================
# 2. DATA CLEANING & PREPROCESSING
# ============================================================

# Convert to numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Replace invalid values
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Median imputation
X = X.fillna(X.median())

# Remove zero variance genes
X = X.loc[:, X.var() > 0]

# RNA-Seq log transformation
X = np.log1p(X)

X = X.values

print("After Cleaning:", X.shape)

# ============================================================
# 3. TAIL-ROBUST QUANTILE NORMALIZATION (TRQN)
# ============================================================

def TRQN(X):
    """
    Converts distribution into Gaussian form
    and clips extreme tails for robustness.
    """

    X_q = quantile_transform(
        X,
        n_quantiles=min(1000, X.shape[0]),
        output_distribution='normal'
    )

    lower = np.percentile(X_q, 1, axis=0)
    upper = np.percentile(X_q, 99, axis=0)

    X_trqn = np.clip(X_q, lower, upper)

    return np.nan_to_num(X_trqn)


X_trqn = TRQN(X)

print("After TRQN:", X_trqn.shape)

# ============================================================
# 4. ADAPTIVE KERNEL PCA (AKPCA)
# ============================================================

# Estimate adaptive kernel width
pairwise_dist = pairwise_distances(X_trqn)
sigma = np.median(pairwise_dist)

adaptive_gamma = 1 / (2 * sigma**2)
gamma_value = 1 / X_trqn.shape[1]

print("Adaptive Gamma:", adaptive_gamma)

# Kernel PCA transformation
kpca = KernelPCA(
    n_components=100,
    kernel='rbf',
    gamma=gamma_value
)

X_kpca = kpca.fit_transform(X_trqn)

print("KPCA Output:", X_kpca.shape)

# ============================================================
# 5. MUTUAL INFORMATION FEATURE SELECTION
# ============================================================

mi_scores = mutual_info_classif(X_kpca, y)

n_feat = 50                     # Top informative components
top_features = np.argsort(mi_scores)[-n_feat:]

X_selected = X_kpca[:, top_features]

selected_gene = gene_names[top_features]

print("After MI Selection:", X_selected.shape)

# ============================================================
# 6. CORRELATION MATRIX VISUALIZATION
# ============================================================

df_sel = pd.DataFrame(
    X_selected[:, :8],
    columns=selected_gene[:8]
)

corr_matrix = df_sel.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    cmap='jet',
    annot=True,
    fmt=".2f",
    square=True
)

plt.tight_layout()
plt.savefig("Result/correlation_mat.jpg", dpi=600)
plt.show()

# ============================================================
# 7. DATA BALANCING USING SMOTE
# ============================================================

print("Before SMOTE:", Counter(y))

smote = SMOTE(
    sampling_strategy='auto',
    random_state=42,
    k_neighbors=4
)

X_selected, y = smote.fit_resample(X_selected, y)

print("After SMOTE:", Counter(y))

# ============================================================
# 8. DATA AUGMENTATION (GAUSSIAN NOISE)
# ============================================================

def augment_noise(X, y, noise_factor=0.0002, copies=10):

    X_aug = [X]
    y_aug = [y]

    for _ in range(copies):
        noise = np.random.normal(0, noise_factor, X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)

    return np.vstack(X_aug), np.hstack(y_aug)


X_selected, y = augment_noise(X_selected, y)

print("After Augmentation:", X_selected.shape)

# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

num_classes = len(np.unique(y))

y_cat = to_categorical(y, num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X_selected,
    y_cat,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ============================================================
# 10. QKAN CUSTOM LAYER
# ============================================================

class QKANLayer(tf.keras.layers.Layer):

    def __init__(self, units, **kwargs):
        super(QKANLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):

        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )

    def call(self, x):

        z = tf.matmul(x, self.W) + self.b

        # Quantum-inspired basis functions
        q1 = tf.sin(z)
        q2 = tf.cos(z)
        q3 = tf.tanh(z)

        return tf.concat([q1, q2, q3], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# ============================================================
# 11. QKAN MODEL ARCHITECTURE
# ============================================================

inputs = Input(shape=(X_train.shape[1],))

x = QKANLayer(256)(inputs)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)

x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)

x = Dense(128, activation='relu')(x)

outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=1e-4,
    weight_decay=1e-4
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# 12. MODEL TRAINING
# ============================================================

if istrain:

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16
    )

    model.save("Model/QKAN_model.keras")
    np.save("Model/training_history.npy", history.history)

    print("Model Saved Successfully!")

# ============================================================
# 13. TRAINING CURVE VISUALIZATION
# ============================================================

hist = np.load("Model/training_history.npy",
               allow_pickle=True).item()

epochs_full = np.arange(1, 101)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# Accuracy
ax1.plot(epochs_full, hist['accuracy'], label="Train Acc")
ax1.plot(epochs_full, hist['val_accuracy'], label="Val Acc")
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0, 1.05)

# Loss
ax2.plot(epochs_full, hist['loss'], '--', label="Train Loss")
ax2.plot(epochs_full, hist['val_loss'], '--', label="Val Loss")
ax2.set_ylabel("Loss")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2)

plt.tight_layout()
plt.savefig("Result/training_.jpg", dpi=600)
plt.show()

# ============================================================
# 14. LOAD TRAINED MODEL
# ============================================================

loaded_model = tf.keras.models.load_model(
    "Model/QKAN_model.keras",
    custom_objects={"QKANLayer": QKANLayer}
)

print("Model Loaded Successfully!")

# ============================================================
# 15. TESTING & PERFORMANCE EVALUATION
# ============================================================

y_pred_prob = loaded_model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

import Performance
Performance.main(np.argmax(y_test, axis=1), y_pred)