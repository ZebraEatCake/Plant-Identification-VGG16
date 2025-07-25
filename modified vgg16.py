import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score, f1_score

# Data input
data_dir   = "/kaggle/input/plantdisease-otsucanny/Dataset (disease)_otsucanny"
batch_size = 64
img_height = 224
img_width  = 224

full_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="int",
    shuffle=True,
)

# Get class names
class_names = full_dataset.class_names
print("Class names:", class_names)

# Split dataset
total_batches = tf.data.experimental.cardinality(full_dataset).numpy()
train_size = int(0.7 * total_batches)
val_size   = int(0.2 * total_batches)
test_size  = total_batches - train_size - val_size

train_ds = full_dataset.take(train_size)
val_ds   = full_dataset.skip(train_size).take(val_size)
test_ds  = full_dataset.skip(train_size + val_size)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.prefetch(buffer_size=AUTOTUNE)

# Model architecture
l2 = regularizers.l2(0.001)

model = Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),

    # Block 1
    layers.Conv2D(32, (3, 3), padding="same", activation="relu", kernel_regularizer=l2),
    layers.Conv2D(32, (3, 3), padding="same", activation="relu", kernel_regularizer=l2),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=l2),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=l2),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=l2),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=l2),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=l2),
    layers.MaxPooling2D((2, 2)),

    # Block 4
    layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=l2),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=l2),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=l2),
    layers.MaxPooling2D((2, 2)),

    # Block 5
    layers.Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=l2),
    layers.Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=l2),
    layers.Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=l2),
    layers.MaxPooling2D((2, 2)),

    # Classifier Head
    layers.GlobalAveragePooling2D(),
    layers.Dense(32, activation="relu", kernel_regularizer=l2),
    layers.Dropout(0.2),
    layers.Dense(3, activation="softmax", kernel_regularizer=l2)
])


# Optimizer
initial_lr = 0.007                         
optimizer = tf.keras.optimizers.SGD(
    learning_rate=initial_lr, 
    momentum=0.9, 
    nesterov= False 
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.9,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

model.compile(
    optimizer=optimizer,                         
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
model.summary()

# Training
epochs  = 100
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks,
)

# Testing
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n Test Accuracy: {test_acc:.2%}")

# Plots
acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))
plt.figure(figsize=(8,8))

plt.subplot(1,2,1)
plt.plot(epochs_range, acc,     label='Train Acc')
plt.plot(epochs_range, val_acc, label='Val Acc')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss,     label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.show()

# Confusion matrix and metrics

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    preds = tf.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Confusion matrix with class labels
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Precision, Recall, F1
precision = precision_score(y_true, y_pred, average='weighted')
recall    = recall_score(y_true, y_pred, average='weighted')
f1        = f1_score(y_true, y_pred, average='weighted')

print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 Score:  {f1:.2%}")

# Detailed classification report with class names
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
