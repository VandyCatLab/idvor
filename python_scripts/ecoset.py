import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PIL import Image


def make_alex_net_v2(
    input_shape=(224, 224, 3),
    output_shape=565,
    weights_path=None,
    softmax=False,
    l2_reg=True,
    augment=False,
):
    inputs = tf.keras.Input(shape=input_shape)

    if augment:
        # Add random flip
        x = tf.keras.layers.RandomFlip("horizontal")(inputs)

        # Add random crop
        x = tf.keras.layers.RandomCrop(224, 224)(x)

        x = layers.Conv2D(
            64,
            (11, 11),
            strides=(4, 4),
            padding="valid",
            activation="relu",
            name="conv1",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005) if l2_reg else None,
        )(x)
    else:
        x = layers.Conv2D(
            64,
            (11, 11),
            strides=(4, 4),
            padding="valid",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005) if l2_reg else None,
            name="conv1",
        )(inputs)

    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), name="pool1")(x)

    x = layers.Conv2D(
        192,
        (5, 5),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.0005) if l2_reg else None,
        name="conv2",
    )(x)
    x = layers.MaxPool2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="pool2",
    )(x)

    x = layers.Conv2D(
        384,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.0005) if l2_reg else None,
        name="conv3",
    )(x)
    x = layers.Conv2D(
        384,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.0005) if l2_reg else None,
        name="conv4",
    )(x)
    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.0005) if l2_reg else None,
        name="conv5",
    )(x)
    x = layers.MaxPool2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="pool5",
    )(x)

    x = layers.Conv2D(
        4096,
        (5, 5),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.0005) if l2_reg else None,
        name="fc6",
    )(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(
        4096,
        (1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.0005) if l2_reg else None,
        name="fc7",
    )(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(
        output_shape,
        (1, 1),
        padding="same",
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005) if l2_reg else None,
        name="fc8",
    )(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)

    if softmax:
        x = layers.Softmax()(x)

    model = tf.keras.Model(inputs, x)

    if weights_path is not None:
        load_alexnet_weights(model, weights_path)

    return model


def make_vNet(
    input_shape=(3, 128, 128),
    output_shape=565,
    weights_path=None,
    softmax=False,
    data_format="channels_first",
):
    normAxis = -3 if data_format == "channels_first" else -1

    inputs = tf.keras.Input(shape=input_shape)

    # Layer 1 - V1
    x = layers.Conv2D(
        128,
        (7, 7),
        name="conv_l1",
        padding="same",
        use_bias=False,
        data_format=data_format,
    )(inputs)
    x = layers.GroupNormalization(axis=normAxis, name="groupNorm_l1")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 2 - V2
    x = layers.Conv2D(
        128,
        (7, 7),
        name="conv_l2",
        padding="same",
        use_bias=False,
        data_format=data_format,
    )(x)
    x = layers.GroupNormalization(axis=normAxis, name="groupNorm_l2")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 3 - V3
    x = layers.MaxPool2D(pool_size=(2, 2), padding="same", data_format=data_format)(x)
    x = layers.Conv2D(
        256,
        (5, 5),
        name="conv_l3",
        padding="same",
        use_bias=False,
        data_format=data_format,
    )(x)
    x = layers.GroupNormalization(axis=normAxis, name="groupNorm_l3")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 4 - hV4
    x = layers.MaxPool2D(pool_size=(2, 2), padding="same", data_format=data_format)(x)
    x = layers.Conv2D(
        256,
        (5, 5),
        name="conv_l4",
        padding="same",
        use_bias=False,
        data_format=data_format,
    )(x)
    x = layers.GroupNormalization(axis=normAxis, name="groupNorm_l4")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 5 - LO
    x = layers.Conv2D(
        512,
        (3, 3),
        name="conv_l5",
        padding="same",
        use_bias=False,
        data_format=data_format,
    )(x)
    x = layers.GroupNormalization(axis=normAxis, name="groupNorm_l5")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 6 - TO
    x = layers.Conv2D(
        512,
        (3, 3),
        name="conv_l6",
        padding="same",
        use_bias=False,
        data_format=data_format,
    )(x)
    x = layers.GroupNormalization(axis=normAxis, name="groupNorm_l6")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 7 - pFUS
    x = layers.MaxPool2D(pool_size=(2, 2), padding="same", data_format=data_format)(x)
    x = layers.Conv2D(
        1024,
        (3, 3),
        name="conv_l7",
        padding="same",
        use_bias=False,
        data_format=data_format,
    )(x)
    x = layers.GroupNormalization(axis=normAxis, name="groupNorm_l7")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 8 - mFUS
    x = layers.MaxPool2D(pool_size=(2, 2), padding="same", data_format=data_format)(x)
    x = layers.Conv2D(
        1024,
        (3, 3),
        name="conv_l8",
        padding="same",
        use_bias=False,
        data_format=data_format,
    )(x)
    x = layers.GroupNormalization(axis=normAxis, name="groupNorm_l8")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 9
    x = layers.MaxPool2D(pool_size=(2, 2), padding="same", data_format=data_format)(x)
    x = layers.Conv2D(
        2048,
        (1, 1),
        name="conv_l9",
        padding="same",
        use_bias=False,
        data_format=data_format,
    )(x)
    x = layers.GroupNormalization(axis=normAxis, name="groupNorm_l9")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Layer 10
    x = layers.MaxPool2D(pool_size=(2, 2), padding="same", data_format=data_format)(x)
    x = layers.Conv2D(
        2048,
        (1, 1),
        name="conv_l10",
        padding="same",
        use_bias=False,
        data_format=data_format,
    )(x)
    x = layers.GroupNormalization(axis=normAxis, name="groupNorm_l10")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.ReLU()(x)

    # Readout
    x = layers.GlobalAveragePooling2D(data_format=data_format)(x)
    x = layers.Flatten(data_format=data_format)(x)
    x = layers.Dense(output_shape, name="readout")(x)

    if softmax:
        x = layers.Softmax()(x)

    model = tf.keras.Model(inputs, x)

    if weights_path is not None:
        load_vNet_weights(model, weights_path)

    return model


def load_alexnet_weights(model, weights_path):
    """Loads the weights into a model from a file in place."""
    reader = tf.train.load_checkpoint(weights_path)
    dtypes = reader.get_variable_to_dtype_map()

    # Loop through layers of the model
    for layer in model.layers:
        # Check if the layer is a conv layer
        if isinstance(layer, layers.Conv2D):
            weightKeys = [
                key
                for key in dtypes.keys()
                if layer.name in key and "Momentum" not in key
            ]
            weightKeys.sort()
            # Get checkpoint weights
            weights = reader.get_tensor(weightKeys[1])
            bias = reader.get_tensor(weightKeys[0])

            # Set weights and biases
            layer.set_weights([weights, bias])

    print(f"Weights from {weights_path} loaded successfully.")


def load_vNet_weights(model, weights_path):
    """Loads the weights into a model from a file in place."""
    reader = tf.train.load_checkpoint(weights_path)
    dtypes = reader.get_variable_to_dtype_map()

    for layer in model.layers:
        # Check if the layer is a conv layer
        if isinstance(layer, layers.Conv2D):
            # Get layer number mapped to the checkpoint
            layerN = str(int(layer.name.split("_")[-1].replace("l", "")) - 1)
            weightKeys = [
                key
                for key in dtypes.keys()
                if "l" + layerN in key and "Adam" not in key and "GroupNorm" not in key
            ]

            # Set weights
            layer.set_weights([reader.get_tensor(weightKeys[0])])

        elif isinstance(layer, layers.GroupNormalization):
            # Get layer number mapped to the checkpoint
            layerN = str(int(layer.name.split("_")[-1].replace("l", "")) - 1)
            weightKeys = [
                key
                for key in dtypes.keys()
                if "l" + layerN in key and "Adam" not in key and "GroupNorm" in key
            ]
            weightKeys.sort()

            # Set weights
            beta = reader.get_tensor(weightKeys[0])
            gamma = reader.get_tensor(weightKeys[1])
            layer.set_weights([gamma, beta])
        elif isinstance(layer, layers.Dense):
            # Get the readout data
            weightKeys = [
                key for key in dtypes.keys() if "readout" in key and "Adam" not in key
            ]
            weightKeys.sort()

            # Set weights
            weights = reader.get_tensor(weightKeys[0])
            biases = reader.get_tensor(weightKeys[1])
            layer.set_weights([weights, biases])

    print(f"Weights from {weights_path} loaded successfully.")


def preprocess_alexnet(image):
    """Preprocesses an image for the AlexNet model."""
    # Convert image to rgb if its directly from pillow
    if isinstance(image, Image.Image):
        image = image.convert("RGB")

    # Change to float32
    image = tf.cast(image, tf.float32)

    # Aspect ratio preserving resize
    image = tf.keras.preprocessing.image.smart_resize(image, (224, 224))

    # Center features
    image = 2 * (image / 255 - 0.5)

    # Expand dim
    if len(image.shape) == 3:
        image = tf.expand_dims(image, 0)

    return image


def preprocess_vNet(image):
    """Preprocesses an image for the vNet model."""
    # Convert image to rgb if its directly from pillow
    if isinstance(image, Image.Image):
        image = image.convert("RGB")

    # Change to float32
    image = tf.cast(image, tf.float32)

    # Aspect ratio preserving resize
    image = tf.keras.preprocessing.image.smart_resize(image, (128, 128))

    # Center features
    image = 2 * (image / 255 - 0.5)

    # Expand dim
    if len(image.shape) == 3:
        image = tf.expand_dims(image, 0)

    # Transpose to channel first format
    image = np.transpose(image, (0, 3, 1, 2))

    return image


if __name__ == "__main__":
    imgPath = "images/allBirds/test/n01604330_17035.JPEG"

    vNet = make_vNet(
        weights_path="./models/vNET/ecoset_training_seeds_01_to_10/training_seed_01/model.ckpt_epoch79",
        softmax=True,
        input_shape=(3, 128, 128),
        data_format="channels_first",
    )

    def _parse_image(x, image_size=(224, 224), data_format="channels_last"):
        # Decode image
        x = tf.io.read_file(x)
        x = tf.io.decode_image(x, channels=3)

        # Add batch channel
        x = tf.expand_dims(x, 0)

        # Cast to float
        x = tf.cast(x, tf.float32)

        # Resize
        x = tf.keras.preprocessing.image.smart_resize(x, image_size)

        # Center features
        x = 2 * (x / 255 - 0.5)

        # Change to channel first format
        if data_format == "channels_first":
            x = tf.transpose(x, (0, 3, 1, 2))

        return x

    # Load a test image
    img = _parse_image(
        imgPath,
        image_size=(128, 128),
        data_format="channels_first",
    )

    # Predict
    vNetPreds = vNet.predict(img)

    # Top 5
    vNetPreds = tf.nn.top_k(vNetPreds, k=5)

    del vNet
    alexNet = make_alex_net_v2(
        weights_path="./models/AlexNet/ecoset_training_seeds_01_to_10/training_seed_01/model.ckpt_epoch89",
        softmax=True,
    )

    # Load a test image
    img = _parse_image(
        imgPath,
        image_size=(224, 224),
        data_format="channels_last",
    )

    # Predict
    alexNetPreds = alexNet.predict(img)

    # Top 5
    alexNetPreds = tf.nn.top_k(alexNetPreds, k=5)
