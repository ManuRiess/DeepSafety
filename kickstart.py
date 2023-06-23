# This is a script for training a neural network with transfer learning on traffic sign recognition / classification

# The script is an adaptation from:
# https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub

# The data to train and validate the model can be downloaded here:
# https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/download

# The code is part of the applied deep learning lecture at RWU
# Contact: Mark.Schutera@gmail.com or mark.schutera@kit.edu


# //////////////////////////////////////// Setup

import numpy as np
import time

import tensorflow as tf
import tensorflow_hub as hub  # conda install -c conda-forge tensorflow-hub
from tensorflow.keras.applications import EfficientNetB0

import matplotlib.pyplot as plt

import datetime
from tensorboard import program


# //////////////////////////////////////// Download Backbone in headless mode
# Meaning these are pure feature extractors

mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
inception_v3 = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

# Check GPU
print(tf.test.is_built_with_cuda())



def gpu_init():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.debugging.enable_traceback_filtering()
    # tf.debugging.set_log_device_placement(True)
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                # tf.config.experimental.set_virtual_device_configuration(
                #     gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1000)])
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def main():
    batch_size = 32
    img_height = 96
    img_width = 96
    # img_height = 224
    # img_width = 224

    feature_extractor_model = (
        # inception_v3  # @param ["mobilenet_v2", "inception_v3"] choose wisely
        EfficientNetB0(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
    )


    # //////////////////////////////////////// Data data data
    # The data to train and validate the model can be downloaded here:
    # https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/download

    # store it to a local folder which you need to define here, for now we only care about the Train data part:
    data_root = "./dataset/Train/"


    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_root,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_root,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    aug = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=False,  # Not allowed as a flipped sign has a different meaning!
        vertical_flip=False,  # Same here...
        validation_split=0.2,
        fill_mode="nearest")

    train_ds_augmented = aug.flow_from_directory(
        data_root,
        subset="training",
        seed=123,
        save_to_dir='./dataset/TrainAug',
        target_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds_augmented = aug.flow_from_directory(
        data_root,
        subset="validation",
        seed=123,
        save_to_dir='./dataset/TrainAug',
        target_size=(img_height, img_width),
        batch_size=batch_size,
    )

    # check whether all your classes have been loaded correctly @class_names ['0' '1' '10' '11' '12']
    class_names = np.array(train_ds.class_names)
    print(class_names)

    # Normalization is done by IDG itself
    # Preprocessing as the tensorflow hub models expect images as float inputs [0,1]
    # normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    # train_ds_augmented = train_ds_augmented.map(
    #     lambda x, y: (normalization_layer(x), y)
    # )  # Where x—images, y—labels.
    # val_ds_augmented = val_ds_augmented.map(
    #     lambda x, y: (normalization_layer(x), y)
    # )  # Where x—images, y—labels.


    # Then we set up prefetching will just smooth your data loader pipeline
    # AUTOTUNE = tf.data.AUTOTUNE
    # train_ds_augmented = train_ds_augmented.cache().prefetch(buffer_size=AUTOTUNE)
    # # val_ds_augmented = val_ds_augmented.cache().prefetch(buffer_size=AUTOTUNE)

    # //////////////////////////////////////// Preparing the model or heating up the coffee machine

    # freeze the feature extractor
    # feature_extractor_layer = hub.KerasLayer(
    #     feature_extractor_model , input_shape=(img_height, img_width, 3), trainable=False
    # )
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    feature_extractor_layer = feature_extractor_model(inputs, training=False)

    # attach the head fitting the current traffic sign classification task
    # the head is a pure fully connected output layer
    num_classes = len(class_names)

    x = tf.keras.layers.GlobalAveragePooling2D()(feature_extractor_layer)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.summary()  # (in case you care how the whole thing looks now)

    # //////////////////////////////////////// Training or wild hand waving on caffeine

    # This starts tensorboard to you can check how your training is progressing
    # Helping you with tracking your training, resort to tensorboard, which can be accessed via the browser
    tracking_address = "./logs"
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

    # This is stuff you are free to play around with
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["acc"],
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )  # Enable histogram computation for every epoch.

    # Add checkpoint every 10 Epochs
    checkpoint_filepath = 'train_checkpoint.h5'

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor='val_loss',
        save_weights_only=False,
        save_best_only=False,
        save_freq='epoch',
        period=10  # Save the model every 10 epochs
    )

    NUM_EPOCHS = 200     # This is probably not enough

    history = model.fit(
        train_ds_augmented,
        validation_data=val_ds_augmented,
        epochs=NUM_EPOCHS,
        callbacks=[tensorboard_callback, checkpoint_callback],
    )

    # Save your model for later use. Early enough you should think about a model versioning system
    # and which information you will need to link with the model when doing so
    t = time.time()

    export_path = "./tmp/saved_models/{}".format(int(t))
    model.save(export_path)
    print("Saved model to " + export_path)

    # Plot history (else see tensorboard)
    training_loss = history.history['loss']
    training_accuracy = history.history['accuracy']

    # Plot the training history
    plt.figure(figsize=(8, 6))
    plt.plot(training_loss, label='Training Loss')
    plt.plot(training_accuracy, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Training History')
    plt.legend()
    plt.savefig('training_results' + str(int(t)) + '.jpg', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    gpu_init()
    main()
