import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

#Current method of loading and normallizing the image using Pillow
def load_image(image_path, label):
    image_path = image_path.decode()

    image = Image.open(image_path)
    image = np.array(image)
    image = image.astype(np.float32) / 255.0

    label = np.int32(label)
    return image, label

def tf_load_image(image_path, label):
    image, label = tf.numpy_function(
        load_image,
        [image_path, label],
        [tf.float32, tf.int32]
    )

    image.set_shape((96, 96, 3))
    label.set_shape(())

    return image, label

#Creates the Dataset for training and testing
def make_dataset(frame, batch=64, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            frame["image_path"].values,
            frame["label"].values.astype(np.int32)
        )
    )

    if shuffle:
        dataset = dataset.shuffle(len(frame))

    dataset = dataset.map(
        tf_load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset.batch(batch).prefetch(tf.data.AUTOTUNE)

def make_autoencoder_dataset(frame, batch=64, shuffle=False):
    dataset = make_dataset(frame, batch=batch, shuffle=shuffle)
 
    dataset = dataset.map(
        lambda images, labels: (
            images,
            {
                "classification": labels,
                "reconstruction": images
            }
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
 
    return dataset.prefetch(tf.data.AUTOTUNE)

def model_builder(input_shape, autoencoder=False):
    #Plain CNN (unchanged)
    if not autoencoder:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
 
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
 
            tf.keras.layers.Flatten(),
 
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
 
        return model
 
    #CNN with autoencoding: identical encoder (same layers, valid padding) as the plain CNN
    inputs = tf.keras.layers.Input(shape=input_shape)
 
    encoded = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    encoded = tf.keras.layers.MaxPooling2D((2, 2))(encoded)
 
    encoded = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(encoded)
    encoded = tf.keras.layers.MaxPooling2D((2, 2))(encoded)
 
    #Classification head (same as plain CNN)
    classified = tf.keras.layers.Flatten()(encoded)
    classified = tf.keras.layers.Dense(64, activation="relu")(classified)
    classification_output = tf.keras.layers.Dense(
        1, activation="sigmoid", name="classification"
    )(classified)
 
    #Decoder: 22x22 -> 44x44 -> 88x88, then resized back to the 96x96 input size
    decoded = tf.keras.layers.UpSampling2D((2, 2))(encoded)
    decoded = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(decoded)
    decoded = tf.keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = tf.keras.layers.Conv2D(3, (3, 3), padding="same", activation="sigmoid")(decoded)
    reconstruction_output = tf.keras.layers.Resizing(
        input_shape[0], input_shape[1], name="reconstruction"
    )(decoded)
 
    return tf.keras.Model(
        inputs=inputs,
        outputs={
            "classification": classification_output,
            "reconstruction": reconstruction_output
        }
    )


#Boolean variable to either train in runtime or load models
USE_PRETRAINED = True

#Base path is indicating where the files are, change as necessary for your data location
base_path = "..\\..\\histopathologic-cancer-detection\\"
df = pd.read_csv(base_path + "train_labels.csv")

train_path = base_path + "train\\"
test_path = base_path + "test\\"

#Allows DataFrame to have an easily accessible path for all .tif files
df["image_path"] = train_path + df["id"] + ".tif"

#Splitting with random state set
train_df, validation_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

train_dataset = make_dataset(train_df, shuffle=True)
validation_dataset = make_dataset(validation_df)

train_ae_dataset = make_autoencoder_dataset(train_df, shuffle=True)
validation_ae_dataset = make_autoencoder_dataset(validation_df)

def make_metrics():
    return [
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]

# CNN
if USE_PRETRAINED:
    model = tf.keras.models.load_model("cnn.keras")
    model.summary()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=make_metrics())
else:
    model = model_builder((96, 96, 3))
    model.summary()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=make_metrics())
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=5)
    model.save("cnn.keras")

results = model.evaluate(validation_dataset, return_dict=True)
print(results)

p, r = results["precision"], results["recall"]
f1 = 2 * p * r / (p + r)

print(f"Precision: {p:.2f}\nRecall: {r:.2f}\nF1-Score: {f1:.2f}")

y_true = validation_df["label"].values
y_prob = model.predict(validation_dataset).ravel()
print("CNN AUC: ", roc_auc_score(y_true, y_prob))

# CNN with Autoencoding
if USE_PRETRAINED:
    model = tf.keras.models.load_model("cnn_ae.keras")
    model.summary()
    model.compile(
        optimizer="adam",
        loss={
            "classification": "binary_crossentropy",
            "reconstruction": "mse"
        },
        loss_weights={
            "classification": 1.0,
            "reconstruction": 1.0
        },
        metrics={"classification": make_metrics()}
    )
else:
    model = model_builder((96, 96, 3), autoencoder=True)
    model.summary()
    model.compile(
        optimizer="adam",
        loss={
            "classification": "binary_crossentropy",
            "reconstruction": "mse"
        },
        loss_weights={
            "classification": 1.0,
            "reconstruction": 1.0
        },
        metrics={"classification": make_metrics()}
    )
    history = model.fit(train_ae_dataset, validation_data=validation_ae_dataset, epochs=5)
    model.save("cnn_ae.keras")

results = model.evaluate(validation_ae_dataset, return_dict=True)
print(results)

p, r = results["classification_precision"], results["classification_recall"]
f1 = 2 * p * r / (p + r)

print(f"Precision: {p:.2f}\nRecall: {r:.2f}\nF1-Score: {f1:.2f}")

#Classification-only view of the trained model (skips building the reconstructions)
classifier = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer("classification").output
)

y_true = validation_df["label"].values
y_prob = classifier.predict(validation_dataset).ravel()
print("CNN with Autoencoding AUC: ", roc_auc_score(y_true, y_prob))

#Testing and Result Exporting
test_labels = pd.read_csv(base_path + "sample_submission.csv")
test_labels["image_path"] = test_path + test_labels["id"] + ".tif"
test_labels["label"] = 0

test_dataset = make_dataset(test_labels, batch=32)

test_labels["label"] = classifier.predict(test_dataset).ravel()
test_labels[["id", "label"]].to_csv("submission.csv", index=False)
