# This is the seed for your validation pipeline. It will allow you to load a model and run it on data from a directory.

# //////////////////////////////////////// Setup

import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def format_string2max(text, max_len):
    out_str = ""
    dif = 0
    for i, letter in enumerate(text):
        if ((i-dif) % max_len == 0) and (i != 0):
            out_str += '\n'
        out_str += letter
        if letter == '\n':
            dif += i
    return out_str


def main():
    # //////////////////////////////////////// Load model
    model_name = "v1_vacc_0.9585"
    import_path = "./trained_models/{}".format(str(model_name))
    model = tf.keras.models.load_model(import_path)

    # //////////////////////////////////////// Load data
    # You will need to unzip the respective batch folders.
    # Obviously Batch_0 is not sufficient for testing as you will soon find out.
    use_test_data = True
    batch_nr = 2
    if use_test_data:
        test_data_root = "./dataset/Test"
    else:
        test_data_root = "./safetyBatches/Batch_" + str(batch_nr) + "/"
    # train_data_root = "./dataset/Train/"
    train_data_root = "./dataset/Train/"

    batch_size = 32
    img_height = 224
    img_width = 224

    train_ds = tf.keras.utils.image_dataset_from_directory(train_data_root)
    # Get information on your train classes
    train_class_names = np.array(train_ds.class_names)
    print("Train Classes available: ", train_class_names)


    # Import lables:
    csv_path = test_data_root + '_label.csv'
    df = pd.read_csv(filepath_or_buffer=csv_path, delimiter=',')
    labels = df['ClassId'].tolist()

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=test_data_root + '/',
        labels=labels,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False,
    )

    files = test_ds.file_paths
    # Get information on your val classes
    class_names = np.array(test_ds.class_names)
    print("Val Classes available: ", class_names)

    if not use_test_data:
        # get the ground truth labels
        test_labels = np.concatenate([y for x, y in test_ds], axis=0)
        # Mapping test labels to the folder names instead of the index
        for i in range(0, len(test_labels)):
            test_labels[i] = int(class_names[test_labels[i]])
    else:
        test_labels = labels

    # Remember that we had some preprocessing before our training this needs to be repeated here
    # Preprocessing as the tensorflow hub models expect images as float inputs [0,1]
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    test_ds = test_ds.map(
        lambda x, y: (normalization_layer(x), y)
    )  # Where x—images, y—labels.

    # //////////////////////////////////////// Inference.
    predictions = model.predict(test_ds)
    predictions = np.argmax(predictions, axis=1)

    # Mapping the prediction class id based on the folder names
    for i in range(0, len(predictions)):
        predictions[i] = int(train_class_names[predictions[i]])

    print("Predictions: ", predictions)
    print("Ground truth: ", test_labels)

    # //////////////////////////////////////// Let the validation begin
    # Probably you will want to at least migrate these to another script or class when this grows..
    def accuracy(predictions, test_labels):
        metric = tf.keras.metrics.Accuracy()
        metric.update_state(predictions, test_labels)
        return metric.result().numpy()

    def mAP(pred, gt):
        metric = tfr.keras.metrics.MeanAveragePrecisionMetric()
        metric.update_state(pred, gt)
        return metric.result().numpy()

    print("Accuracy: ", accuracy(predictions, test_labels))

    class_names_df = pd.read_csv("./signnames.csv", index_col="ClassId")
    print(
        f'Class ID: {predictions[0]} Class Name: {class_names_df["SignName"][predictions[0]]}'
    )

    # Show wrongly classified data
    idx_wrong_class = [i for i in range(0, len(predictions)) if predictions[i] != test_labels[i]]

    # cut to0 much data
    if(len(idx_wrong_class)>=100):
        idx_wrong_class_cut = idx_wrong_class[:100]
    else:
        idx_wrong_class_cut = idx_wrong_class
    numbers_to_display = len(idx_wrong_class_cut)
    num_cells = math.ceil(math.sqrt(numbers_to_display))
    plt.figure(figsize=(15, 15))

    for i in range(0, len(idx_wrong_class_cut)):
        idx = idx_wrong_class_cut[i]
        predicted_label = f'{predictions[idx]} - {class_names_df["SignName"][predictions[idx]]}\n' \
                          f'GT: {test_labels[idx]} - {class_names_df["SignName"][test_labels[idx]]}'
        predicted_label = format_string2max(predicted_label, 25)
        plt.grid(False)
        # color_map = 'Greens' if predictions[idx] == test_labels[idx] else 'Reds'
        # plt.set_cmap(color_map)
        plt.subplot(num_cells, num_cells, i + 1)
        img = mpimg.imread(files[idx])
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        font = {'color': 'green', 'weight': 'normal', 'size': 13} if predictions[idx] == test_labels[idx] \
            else {'color': 'red', 'weight': 'bold', 'size': 13}
        plt.xlabel(predicted_label,  fontdict=font)

    plt.subplots_adjust(hspace=1, wspace=0.5)
    plt.show()

    # Plot error matrix
    num_cls = len(class_names_df)
    error_matrix_df = pd.DataFrame(np.zeros((num_cls, num_cls)), columns=class_names_df["SignName"].tolist(),)
    for i in range(0, len(idx_wrong_class_cut)):
        idx = idx_wrong_class[i]
        error_matrix_df[test_labels[idx], predictions[idx]] += 1

    # There is more and this should get you started: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
    # However it is not about how many metrics you crank out, it is about whether you find the meaningful ones and report on them.
    # Think about a system on how to decide which metric to go for..

    # You are looking for a great package to generate your reports, let me recommend https://plotly.com/dash/


if __name__ == "__main__":
    main()
