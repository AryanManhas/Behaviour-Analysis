"""
dog_aggression_demo.py
Demo: multisensor dog aggression detection (IMU 1D-CNN + Temp MLP + learned fusion).
Generates synthetic IMU + Temp data, trains models, evaluates, visualizes confidence & accuracy,
and saves models to model_weights/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
import random

# -------------------------
# Config
# -------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

NUM_CLASSES = 3  # 0=Calm,1=Alert,2=Aggressive
CLASS_NAMES = ['Calm', 'Alert', 'Aggressive']

# Synthetic IMU sequence params
SEQ_LEN = 100            # time steps per sample
IMU_CHANNELS = 6         # 3 accel + 3 gyro
SAMPLES_PER_CLASS = 1200 # total samples per class
BATCH_SIZE = 64
EPOCHS = 25
MODEL_DIR = "model_weights"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Synthetic data generation
# -------------------------
def synth_imu_sequence(label, seq_len=SEQ_LEN):
    """
    Generate synthetic 6-channel IMU data.
    Patterns:
      - Calm: low amplitude gaussian noise
      - Alert: oscillatory moderate amplitude
      - Aggressive: baseline + sharp spikes / bursts (lunges)
    Return shape: (seq_len, 6)
    """
    t = np.linspace(0, 1, seq_len)
    # Base noise (all)
    base = np.random.normal(scale=0.02, size=(seq_len, IMU_CHANNELS))
    if label == 0:  # Calm
        # Very smooth slight movement
        drift = np.outer(np.sin(2*np.pi*0.5*t)*0.02, np.ones(IMU_CHANNELS))
        return base + drift
    elif label == 1:  # Alert
        # Higher oscillation: head/ear twitch, tail flick
        osc = np.sin(2*np.pi*5*t)[:, None] * (0.05 + 0.02*np.random.rand(IMU_CHANNELS))
        # add occasional small bursts
        bursts = np.zeros((seq_len, IMU_CHANNELS))
        for _ in range(np.random.randint(1,4)):
            idx = np.random.randint(10, seq_len-10)
            bursts[idx:idx+5] += np.random.normal(0.15, 0.05, size=(5, IMU_CHANNELS))
        return base + osc + bursts
    elif label == 2:  # Aggressive
        # Sudden lunges + high variance shakes
        # baseline bias + spikes at random times
        bias = np.outer(np.ones(seq_len)*0.02, np.ones(IMU_CHANNELS))
        spikes = np.zeros((seq_len, IMU_CHANNELS))
        # Several random spikes in the sequence
        for _ in range(np.random.randint(2,5)):
            idx = np.random.randint(5, seq_len-6)
            magnitude = 0.5 + np.random.rand()*0.5
            # lunge spike shape (sharp)
            spike_shape = np.concatenate([np.linspace(0, magnitude, 2),
                                          np.linspace(magnitude, 0, 3)])
            for ch in range(IMU_CHANNELS):
                spikes[idx:idx+5, ch] += spike_shape * (0.6 + 0.8*np.random.rand())
        jitter = np.random.normal(0, 0.06, size=(seq_len, IMU_CHANNELS))
        return base + bias + spikes + jitter
    else:
        raise ValueError("Unknown label")

def synth_temp_value(label):
    """
    Generate a single temperature reading depending on class.
      - Calm: 36.0 - 37.5
      - Alert: 37.6 - 38.5
      - Aggressive: 38.6 - 39.5
    Returns scalar float.
    """
    if label == 0:
        return 36.0 + np.random.rand() * 1.5
    elif label == 1:
        return 37.6 + np.random.rand() * 0.9
    elif label == 2:
        return 38.6 + np.random.rand() * 0.9
    else:
        raise ValueError("Unknown label")

def generate_dataset(n_per_class=SAMPLES_PER_CLASS):
    X_imu = []
    X_temp = []
    y = []
    for label in range(NUM_CLASSES):
        for _ in range(n_per_class):
            imu_seq = synth_imu_sequence(label)
            temp = synth_temp_value(label)
            X_imu.append(imu_seq.astype(np.float32))
            X_temp.append(np.array([temp], dtype=np.float32))
            y.append(label)
    X_imu = np.array(X_imu)   # shape (N, seq_len, channels)
    X_temp = np.array(X_temp) # shape (N, 1)
    y = np.array(y, dtype=np.int32)
    # Shuffle
    idx = np.random.permutation(len(y))
    return X_imu[idx], X_temp[idx], y[idx]

# -------------------------
# Build models
# -------------------------
def build_imu_model(input_shape=(SEQ_LEN, IMU_CHANNELS)):
    inp = layers.Input(shape=input_shape, name="imu_input")
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    out_feat = layers.Dense(32, activation='relu', name='imu_feature')(x)  # feature vector
    # Optionally add a branch to predict directly (not used for fusion training)
    out_class = layers.Dense(NUM_CLASSES, name='imu_logits')(out_feat)
    model = models.Model(inputs=inp, outputs=[out_feat, out_class], name='imu_model')
    return model

def build_temp_model(input_shape=(1,)):
    inp = layers.Input(shape=input_shape, name='temp_input')
    x = layers.Normalization(axis=-1)(inp)  # will be adapted later
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    out_feat = layers.Dense(8, activation='relu', name='temp_feature')(x)
    out_class = layers.Dense(NUM_CLASSES, name='temp_logits')(out_feat)
    model = models.Model(inputs=inp, outputs=[out_feat, out_class], name='temp_model')
    return model

def build_fusion_model():
    # inputs
    imu_in = layers.Input(shape=(SEQ_LEN, IMU_CHANNELS), name='imu_input_full')
    temp_in = layers.Input(shape=(1,), name='temp_input_full')

    # imu submodel
    imu_model = build_imu_model()
    temp_model = build_temp_model()

    imu_feat, imu_logits = imu_model(imu_in)
    temp_feat, temp_logits = temp_model(temp_in)

    # learned fusion: concatenate features
    fused = layers.Concatenate()([imu_feat, temp_feat])
    f = layers.Dense(64, activation='relu')(fused)
    f = layers.Dropout(0.3)(f)
    f = layers.Dense(32, activation='relu')(f)
    logits = layers.Dense(NUM_CLASSES)(f)  # final logits
    probs = layers.Activation('softmax', name='probs')(logits)

    model = models.Model(inputs=[imu_in, temp_in], outputs=probs, name='fusion_model')
    return model

# -------------------------
# Training + Evaluation
# -------------------------
def prepare_and_train():
    print("Generating synthetic dataset...")
    X_imu, X_temp, y = generate_dataset()
    # Normalize temperature normalization layer will be adapted automatically in temp model
    # Split
    X_imu_train, X_imu_test, X_temp_train, X_temp_test, y_train, y_test = train_test_split(
        X_imu, X_temp, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    # Build fusion model
    fusion_model = build_fusion_model()
    fusion_model.compile(optimizer=optimizers.Adam(1e-3),
                         loss=losses.SparseCategoricalCrossentropy(),
                         metrics=['accuracy'])
    fusion_model.summary()

    # Fit normalization layer for temp submodel inside fusion (we must adapt using the submodel)
    # Find the temp model inside fusion_model
    # Simpler approach: adapt manually using mean/std
    temp_mean = X_temp_train.mean(axis=0)
    temp_std = X_temp_train.std(axis=0) + 1e-8

    # Create simple data generator / dataset
    train_ds = tf.data.Dataset.from_tensor_slices(((X_imu_train, X_temp_train), y_train))
    train_ds = train_ds.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices(((X_imu_test, X_temp_test), y_test)).batch(BATCH_SIZE)

    # Use callbacks
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, 'fusion_best.h5'),
                                  monitor='val_loss', save_best_only=True)
    ]

    # Train
    print("Training fusion model...")
    history = fusion_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb, verbose=2)

    # Evaluate
    test_loss, test_acc = fusion_model.evaluate(val_ds, verbose=0)
    print(f"Test accuracy: {test_acc*100:.2f}%")

    # Save final model
    fusion_model.save(os.path.join(MODEL_DIR, 'fusion_final.h5'))
    print("Models saved to", MODEL_DIR)
    return fusion_model, (X_imu_test, X_temp_test, y_test), history

# -------------------------
# Visualization + Inference
# -------------------------
def plot_results_and_demo(model, X_test_tuple, y_test, history):
    X_imu_test, X_temp_test, y_test = X_test_tuple
    # Compute predictions on test set
    preds = model.predict((X_imu_test, X_temp_test), batch_size=128)
    pred_labels = np.argmax(preds, axis=1)
    acc = np.mean(pred_labels == y_test)
    print(f"Validation Accuracy = {acc*100:.2f}%")

    # Build simple matplotlib dashboard: left shows confidence bar for a sample,
    # right shows accuracy bar across test set.

    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    ax_conf, ax_acc = axs

    # Prepare accuracy bar
    ax_acc.set_title("Model Accuracy")
    ax_acc.set_xlim(0, 100)
    acc_bar = ax_acc.barh([0], [acc*100], color='tab:blue')
    ax_acc.set_yticks([])
    ax_acc.set_xlabel("Accuracy (%)")
    ax_acc.set_xlim(0, 100)
    ax_acc.set_xticks([0, 25, 50, 75, 100])

    # Confidence bar
    ax_conf.set_title("Aggression Confidence")
    ax_conf.set_xlim(0, 100)
    conf_bar = ax_conf.barh([0], [0], color='tab:red')
    ax_conf.set_yticks([])
    ax_conf.set_xlabel("Confidence (%)")
    ax_conf.set_xticks([0, 25, 50, 75, 100])

    plt.tight_layout()
    plt.show()

    # Demo loop: show 12 random test samples sequentially (like live inference)
    n_demos = 12
    demo_indices = np.random.choice(len(y_test), size=n_demos, replace=False)
    for idx in demo_indices:
        imu = X_imu_test[idx:idx+1]
        temp = X_temp_test[idx:idx+1]
        true_label = y_test[idx]
        probs = model.predict((imu, temp), verbose=0)  # shape (1, num_classes)
        top_idx = np.argmax(probs[0])
        confidence = probs[0, top_idx] * 100

        # Update confidence bar
        conf_bar[0].set_width(confidence)
        conf_bar[0].set_color('tab:green' if top_idx==2 else 'tab:orange' if top_idx==1 else 'tab:blue')
        ax_conf.set_title(f"Aggression Confidence - Pred: {CLASS_NAMES[top_idx]} ({confidence:.1f}%) | True: {CLASS_NAMES[true_label]}")
        # Update accuracy bar (static)
        acc_bar[0].set_width(acc*100)
        ax_acc.set_title(f"Model Accuracy = {acc*100:.2f}%")

        plt.pause(1.1)  # pause to simulate live update

    plt.ioff()
    plt.show()

# -------------------------
# Main
# -------------------------
def main():
    model, test_tuple, history = prepare_and_train()
    plot_results_and_demo(model, test_tuple, test_tuple[2], history)
    print("Demo finished. Models are in", MODEL_DIR)
    print("To replace synthetic data with real IMU/temperature CSVs, see comments in the script.")

if __name__ == "__main__":
    main()
