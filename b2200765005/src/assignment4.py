import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

tf.config.list_physical_devices('GPU')

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


import os

def find_image_paths_with_n_images(directory, n=300):
    """
    Find image paths for the first n images within the specified directory for each class that has at least n images.
    
    :param directory: The base directory to search within.
    :param n: The number of images to retrieve from each directory.
    :return: A dictionary with keys as directory names and values as lists of image file paths.
    """
    directory_images = {}
    # Iterate over the subdirectories
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        # Check if the path is a directory
        if os.path.isdir(subdir_path):
            # Get all image file paths in the directory
            image_files = [os.path.join(subdir_path, file) for file in os.listdir(subdir_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
            # If the directory has at least n images, take the first n images
            if len(image_files) >= n:
                directory_images[subdir] = image_files[:n]  # Only take the first n images
    
    return directory_images

# Define the base directory where the images are stored
base_dir = 'dataset/indoorCVPR_09/Images'

# Call the function and store the result
directory_names_with_n_images = find_image_paths_with_n_images(base_dir, n=300)

# Print the directory names
print(directory_names_with_n_images)

for directory, images in directory_names_with_n_images.items():
    print(f"{directory}: {len(images)}")



img_width, img_height = 128, 128  # Modify as per your requirement
batch_size = 32
num_classes = len(directory_names_with_n_images)  

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


import pandas as pd
from sklearn.model_selection import train_test_split


train_df_list, validation_df_list, test_df_list = [], [], []

for folder_name in directory_names_with_n_images:
    folder_path = os.path.join(base_dir, folder_name)
    folder_images = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # Split the data
    folder_train = pd.DataFrame({'filepath': folder_images[:200], 'label': folder_name})
    folder_validation = pd.DataFrame({'filepath': folder_images[200:250], 'label': folder_name})
    folder_test = pd.DataFrame({'filepath': folder_images[250:300], 'label': folder_name})

    # Append to lists
    train_df_list.append(folder_train)
    validation_df_list.append(folder_validation)
    test_df_list.append(folder_test)

# Combine all the splits
train_df = pd.concat(train_df_list).reset_index(drop=True)
validation_df = pd.concat(validation_df_list).reset_index(drop=True)
test_df = pd.concat(test_df_list).reset_index(drop=True)


# Create the ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Use flow_from_dataframe to create the generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col='label',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,
    x_col='filepath',
    y_col='label',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepath',
    y_col='label',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Typically, you don't shuffle the test set
)


def build_model(dropout_rate):

    model_cnn = Sequential([
        # First convolutional layer with more filters and batch normalization
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(128,128,3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Second convolutional layer with increased filters and batch normalization
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Third convolutional layer
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Fourth convolutional layer
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Flattening the 3D output to 1D
        Flatten(),
        
        # Dense layer with more neurons and batch normalization
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output layer with 'softmax' activation for multi-class classification
        Dense(num_classes, activation='softmax')
    ])

    return model_cnn


import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load the VGG16 model, pretrained on ImageNet data, excluding the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze all layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Create a new top (fully connected) layers for your specific classification task
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model_allfreeze = Model(inputs=base_model.input, outputs=predictions)


# Load the VGG16 model, pretrained on ImageNet data, excluding the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze all layers of the base model
for layer in base_model.layers[:-8]:  # Freezes all layers except the last two convolutional blocks
    layer.trainable = False
for layer in base_model.layers[-8:]:
    layer.trainable = True

# Create a new top (fully connected) layers for your specific classification task
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model_first2 = Model(inputs=base_model.input, outputs=predictions)



learning_rates = [0.001 , 0.0001, 0.00001]
batch_sizes = [32, 64]
early_stop = EarlyStopping(monitor='val_loss', patience=3)

from sklearn.metrics import confusion_matrix
import numpy as np


def train_and_evaluate(model, learning_rate, batch_size):
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=30 , # Modify as per your requirement
        callbacks=[early_stop]
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy with LR={learning_rate}, Batch Size={batch_size}: {accuracy*100:.2f}%")

    
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)

    #Now compute the confusion matrix
    cm = confusion_matrix(test_generator.classes, predicted_classes)

    return history, loss, accuracy, cm
 
import os

def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

import os

dropout_results_dir = "dropout_results"
if not os.path.exists(dropout_results_dir):
    os.makedirs(dropout_results_dir)

from matplotlib import pyplot as plt
import seaborn as sns


def plot_results(history, cm, model_name, lr, batch_size):
    dir_name = f"{model_name}_results"
    create_directory(dir_name)

    # Plot and save training history
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy - {model_name} (LR={lr}, Batch={batch_size})')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f"{dir_name}/Accuracy_LR{lr}_Batch{batch_size}.png")
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss - {model_name} (LR={lr}, Batch={batch_size})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f"{dir_name}/Loss_LR{lr}_Batch{batch_size}.png")
    plt.close()

    # Plot and save confusion matrix
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix - {model_name} (LR={lr}, Batch={batch_size})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"{dir_name}/ConfusionMatrix_LR{lr}_Batch{batch_size}.png")
    plt.close()


results = []

dropout_rates = [0.2, 0.3, 0.4, 0.5]
for rate in dropout_rates:
    print(f"\nTraining with dropout rate: {rate}")
    model = build_model(rate)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, validation_data=validation_generator, epochs=30,callbacks=[early_stop])
    val_loss, val_accuracy = model.evaluate(validation_generator)
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Validation Accuracy with dropout {rate}: {val_accuracy*100:.2f}%")
    print(f"Test Accuracy with dropout {rate}: {test_accuracy*100:.2f}%")

    # Save the results
    results_file = os.path.join(dropout_results_dir, f"dropout_{rate}_results.txt")
    with open(results_file, "w") as file:
        file.write(f"Dropout Rate: {rate}\n")
        file.write(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}\n")
        file.write(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}\n")

for lr in learning_rates:
    print(f"\nTraining Custom CNN with LR={lr}, Batch Size={batch_size}")
    model = build_model(0.2)
    history_cnn = train_and_evaluate(model, lr, batch_size)
    
    print(f"\nTraining VGG16-based Model with LR={lr}, Batch Size={batch_size}")
    history_vgg16_allfreeze = train_and_evaluate(model_allfreeze, lr, batch_size)
    history_vgg16_first2 = train_and_evaluate(model_first2, lr, batch_size)

    results.append({'model': 'CNN', 'lr': lr, 'batch_size': batch_size, 'results': history_cnn})
    results.append({'model': 'VGG16_allfreeze', 'lr': lr, 'batch_size': batch_size, 'results': history_vgg16_allfreeze})
    results.append({'model': 'VGG16_first2', 'lr': lr, 'batch_size': batch_size, 'results': history_vgg16_first2})

for batch_size in batch_sizes:

    print(f"\nTraining Custom CNN with LR={lr}, Batch Size={batch_size}")
    model = build_model(0.2)
    history_cnn = train_and_evaluate(model, lr, batch_size)
    
    print(f"\nTraining ALL_FREEZED-VGG16-based Model with LR={lr}, Batch Size={batch_size}")
    history_vgg16_allfreeze = train_and_evaluate(model_allfreeze, lr, batch_size)
    print(f"\nTraining FIRST2LAYER_UNFREEZED_VGG16-based Model with LR={lr}, Batch Size={batch_size}")
    history_vgg16_first2 = train_and_evaluate(model_first2, lr, batch_size)

    results.append({'model': 'CNN', 'lr': lr, 'batch_size': batch_size, 'results': history_cnn})
    results.append({'model': 'VGG16_allfreeze', 'lr': lr, 'batch_size': batch_size, 'results': history_vgg16_allfreeze})
    results.append({'model': 'VGG16_first2', 'lr': lr, 'batch_size': batch_size, 'results': history_vgg16_first2})


for result in results:
    model_name = result['model']
    lr = result['lr']
    batch_size = result['batch_size']
    history, loss, accuracy, cm = result['results']

    print(f"Results for {model_name} with LR={lr} and Batch Size={batch_size}")
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    plot_results(history, cm, model_name, lr, batch_size)

results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def plot_by_learning_rate(results, results_dir):
    accuracies_by_lr = {}
    for result in results:
        lr = result['lr']
        model_name = result['model']
        _, _, accuracy, _ = result['results']

        if lr not in accuracies_by_lr:
            accuracies_by_lr[lr] = {}
        if model_name not in accuracies_by_lr[lr]:
            accuracies_by_lr[lr][model_name] = []

        accuracies_by_lr[lr][model_name].append(accuracy)

    for lr, model_accuracies in accuracies_by_lr.items():
        plt.figure(figsize=(10, 6))
        for model, accuracies in model_accuracies.items():
            plt.plot(accuracies, label=model, linewidth=2, marker='o')
        plt.title(f'Test Accuracies by Learning Rate: {lr}')
        plt.xlabel('Experiment Number')
        plt.ylabel('Test Accuracy')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f'test_results_lr_{lr}.png'))  # Save figure to the 'results' folder
        plt.close()

def plot_by_batch_size(results, results_dir):
    accuracies_by_batch = {}
    for result in results:
        batch_size = result['batch_size']
        model_name = result['model']
        _, _, accuracy, _ = result['results']

        if batch_size not in accuracies_by_batch:
            accuracies_by_batch[batch_size] = {}
        if model_name not in accuracies_by_batch[batch_size]:
            accuracies_by_batch[batch_size][model_name] = []

        accuracies_by_batch[batch_size][model_name].append(accuracy)

    for batch_size, model_accuracies in accuracies_by_batch.items():
        plt.figure(figsize=(10, 6))
        for model, accuracies in model_accuracies.items():
            plt.plot(accuracies, label=model, linewidth=2, marker='o')
        plt.title(f'Test Accuracies by Batch Size: {batch_size}')
        plt.xlabel('Experiment Number')
        plt.ylabel('Test Accuracy')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f'test_results_batch_{batch_size}.png'))  # Save figure to the 'results' folder
        plt.close()

# Call the functions with your results and the directory path
plot_by_learning_rate(results, results_dir)
plot_by_batch_size(results, results_dir)

import tensorflow as tf
import os
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image

def load_images_and_labels(image_paths, img_height, img_width):
    images = []
    labels = []
    
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=(img_height, img_width))
        img = image.img_to_array(img)
        img = tf.image.per_image_standardization(img)  # Normalize the image
        
        label = 1 if 'raccoon' in img_path else 0  # Assuming binary classification (raccoon/not raccoon)
        images.append(img)
        labels.append(label)
    
    return tf.data.Dataset.from_tensor_slices((images, labels))

# Get the image paths
train_image_paths = glob(os.path.join(base_dir, 'train', '*.jpg'))
valid_image_paths = glob(os.path.join(base_dir, 'valid', '*.jpg'))
test_image_paths = glob(os.path.join(base_dir, 'test', '*.jpg'))

img_height=128
img_width=128

# Load and preprocess the images
train_ds = load_images_and_labels(train_image_paths, img_height, img_width).batch(batch_size)
val_ds = load_images_and_labels(valid_image_paths, img_height, img_width).batch(batch_size)
test_ds = load_images_and_labels(test_image_paths, img_height, img_width).batch(batch_size)


from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def build_vgg16_classification_model(input_shape):
    # Load the VGG16 model, pretrained on ImageNet data
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers of the VGG16 model to prevent them from being trained
    for layer in vgg16_model.layers:
        layer.trainable = False

    # Create a new model and add the VGG16 layers
    model = models.Sequential()
    model.add(vgg16_model)

    # Add new layers for your specific task
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='softmax'))  # Use 'sigmoid' for binary classification

    return model


from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def build_vgg16_classification_and_regression_model(input_shape):
    # Load the VGG16 model, pretrained on ImageNet data
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers of the VGG16 model to prevent them from being trained
    for layer in vgg16_model.layers:
        layer.trainable = False

    # Add new layers on top of VGG16 base
    x = layers.Flatten()(vgg16_model.output)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Classification head
    classification_output = layers.Dense(1, activation='softmax', name='classification_head')(x)

    # Regression head (for bounding box prediction)
    # This head outputs 4 values: [x, y, width, height]
    regression_output = layers.Dense(4, activation='linear', name='regression_head')(x)

    # Construct the full model with both heads
    model = models.Model(inputs=vgg16_model.input, outputs=[classification_output, regression_output])

    return model


import os
import matplotlib.pyplot as plt

early_stop_fr = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

# Directory to save validation results
results_dir = 'racoon_valid_results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

batch_sizes = [32, 64]
learning_rates = [0.001, 0.0001]
input_shape = (128, 128, 3)

# Store validation accuracies for plotting
validation_accuracies = {
    'classification': {},
    'regression': {}
}

loss_history = {
    'classification': {'softmax': [], 'combined': []},
    'regression': {'l2': [], 'combined': []}
}


for batch_size in batch_sizes:
    for lr in learning_rates:
        # Classification Model
        classification_model = build_vgg16_classification_model(input_shape)
        classification_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                                     loss='binary_crossentropy',
                                     metrics=['accuracy'])
        classification_history = classification_model.fit(train_ds, epochs=10, batch_size=batch_size, validation_data=val_ds,callbacks=[early_stop_fr])
        
        
        
        loss_history['classification']['softmax'].extend(classification_history.history['loss'])

        
        # Regression Model
        regression_model = build_vgg16_classification_and_regression_model(input_shape)
        regression_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                                 loss={'classification_head': 'binary_crossentropy', 'regression_head': 'mean_squared_error'},
                                 metrics={'classification_head': 'accuracy'})
        regression_history = regression_model.fit(train_ds, epochs=10, batch_size=batch_size, validation_data=val_ds,callbacks=[early_stop_fr])
        
        
        loss_history['regression']['l2'].extend(regression_history.history['regression_head_loss'])
       
        # Calculate combined lossesss
        classification_combined_loss = [x + y for x, y in zip(classification_history.history['loss'], regression_history.history['regression_head_loss'])]
        regression_combined_loss = classification_combined_loss
        loss_history['classification']['combined'].extend(classification_combined_loss)
        loss_history['regression']['combined'].extend(regression_combined_loss)

        val_acc_key = f'bs{batch_size}_lr{lr}'
        validation_accuracies['classification'][val_acc_key] = classification_history.history['val_accuracy']
        validation_accuracies['regression'][val_acc_key] = regression_history.history['val_classification_head_accuracy']


        
        
        for model_type, accuracies in validation_accuracies.items():
            for key, vals in accuracies.items():
                plt.figure()
                plt.plot(vals, label=f'{model_type.capitalize()} Model ({key})')
                plt.xlabel('Epoch')
                plt.ylabel('Validation Accuracy')
                plt.title(f'Validation Accuracy for {model_type.capitalize()} Model ({key})')
                plt.legend()
                plt.savefig(os.path.join(results_dir, f'{model_type}_validation_accuracy_{key}.png'))
                plt.close()

        for model_type, losses in loss_history.items():
            plt.figure(figsize=(10, 4))
            for loss_type, values in losses.items():
                plt.plot(values, label=f'{loss_type.capitalize()} Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{model_type.capitalize()} Model Loss (Batch Size: {batch_size}, LR: {lr})')
            plt.legend()
            plt.savefig(f'{results_dir}/{model_type}_model_loss_{val_acc_key}.png')
            plt.close()

print("All training and plotting complete!")


regression_model = build_vgg16_classification_and_regression_model(input_shape)
regression_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                                 loss={'classification_head': 'binary_crossentropy', 'regression_head': 'mean_squared_error'},
                                 metrics={'classification_head': 'accuracy'})
regression_history = regression_model.fit(train_ds, epochs=10, batch_size=32, validation_data=val_ds,callbacks=[early_stop_fr])

# Iterate through the test dataset and predict bounding boxes
for image_batch, _ in test_ds.take(1):  # take(1) will give you one batch
    # Predict using the regression model
    predictions = regression_model.predict(image_batch)
    bbox_predictions = predictions[1]
 
    # Get the dimensions of the images
    img_width, img_height = image_batch.shape[2], image_batch.shape[1]

    # Convert bounding box predictions to the pixel range
    bbox_predictions[:, 0] *= img_width  # x
    bbox_predictions[:, 1] *= img_height  # y
    bbox_predictions[:, 2] *= img_width  # width
    bbox_predictions[:, 3] *= img_height  # height

    print(bbox_predictions)

from matplotlib import patches


def plot_image_with_boxes(image, boxes):
    # Assuming image is a numpy array of shape (height, width, channels)
    height, width, _ = image.shape
    plt.imshow(image)

    for box in boxes:
        # Scale the bounding box coordinates to the image size
        xmin, ymin, width, height = box
        xmin *= img_width
        width *= img_width
        ymin *= img_height
        height *= img_height

        # Create a Rectangle patch
        # Create a Rectangle patch with transparent facecolor
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=10, edgecolor='r', facecolor='white')



        # Add the patch to the Axes
        plt.gca().add_patch(rect)

    plt.show()

# Example usage:
# Iterate through the test dataset and predict bounding boxes
# Example usage:
# Iterate through the test dataset and predict bounding boxes
for i, (image_batch, _) in enumerate(test_ds.take(1)):  # take(1) will give you one batch
    # Predict using the regression model
    predictions = regression_model.predict(image_batch)
    bbox_predictions = predictions[1]

    # Get the dimensions of the images
    img_width, img_height = image_batch.shape[2], image_batch.shape[1]

    # Convert bounding box predictions to the pixel range
    bbox_predictions[:, 0] *= img_width  # x
    bbox_predictions[:, 1] *= img_height  # y
    bbox_predictions[:, 2] *= img_width  # width
    bbox_predictions[:, 3] *= img_height  # height

    print(bbox_predictions[0, 0])
    print(bbox_predictions[0, 1])
    print(bbox_predictions[0, 2])
    print(bbox_predictions[0, 3])

    # Normalize the image pixel values to [0, 1]
image_batch[0] = (image_batch[0] - np.min(image_batch[0])) / (np.max(image_batch[0]) - np.min(image_batch[0]))

# Plot the image with bounding boxes
plot_image_with_boxes(image_batch[0].numpy(), [(bbox_predictions[0, 0], bbox_predictions[0, 1], bbox_predictions[0, 2], bbox_predictions[0, 3])])


