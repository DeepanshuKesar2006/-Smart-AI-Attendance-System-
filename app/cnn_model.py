from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam


def build_cnn(input_shape=(128, 128, 3), num_classes=3):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D(2, 2),

        # layers.Conv2D(1024, (3, 3), activation="relu", padding="same"),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D(2, 2),

        # GlobalAveragePooling instead of Flatten — this is the key fix.
        # Flatten -> Dense(25088 -> 128) was 3M+ params from a tiny dataset,
        # which is what caused the unstable, exploding gradients.
        layers.GlobalAveragePooling2D(),

        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        # layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),  # lower LR + gradient clipping
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model