# command ในนี้ อิงตาม flowchart นะครับ
#import library
import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# IF have GPU?
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    #Run with GPU
    tf.config.experimental.set_memory_growth(gpu, True)

# Print available GPU list
print(tf.config.list_physical_devices('GPU'))

# Set variable data_dir = 'data'
data_dir = 'data'
# Set variable image_exts = jpeg, jpg, bmp, png
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

# For each folder in data_dir (image_class)
for image_class in os.listdir(data_dir):
    # For each file in image_class folder (image)
    for image in os.listdir(os.path.join(data_dir, image_class)):
        # Set image_path = data_dir/image_class/image
        image_path = os.path.join(data_dir, image_class, image)
        # Try to read image (cv2.imread)
        try:
            # cv2.imread(image_path)
            img = cv2.imread(image_path)
            # if is image loaded successfully?
            if img is None:
                # Print "Cannot read image" and Delete file
                print(f'Cannot read image: {image_path}')
                os.remove(image_path)
                continue
            tip = imghdr.what(image_path)
            # if is image loaded successfully?
            if tip not in image_exts:
                # Print "Image not in ext list" and Delete file
                print('Image not in ext list: {}'.format(image_path))
                os.remove(image_path)
        # Catch Exception
        except Exception as e:
            # Print "Issue with image" and Continue
            print('Issue with image: {}'.format(image_path), e)

# Load dataset with image_dataset
data = tf.keras.utils.image_dataset_from_directory('data', image_size=(256, 256), batch_size=32)
# Normalize dataset (x / 255)
data = data.map(lambda x, y: (x / 255, y))

# split dataset
# 70% for train
# 20% for val
# 10% test
dataset_size = data.cardinality().numpy()
train_size = int(dataset_size * 0.7)
val_size = int(dataset_size * 0.2)
test_size = int(dataset_size * 0.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Build CNN model  
model = Sequential([
    Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    Conv2D(32, (3, 3), 1, activation='relu'),
    MaxPooling2D(),
    Conv2D(16, (3, 3), 1, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model  
model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.summary()

# Train model with 100 epochs
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=100, validation_data=val, callbacks=[tensorboard_callback])

# Plot training and validation loss
plt.figure()
plt.plot(hist.history['loss'], label='Loss', color='teal')
plt.plot(hist.history['val_loss'], label='Val Loss', color='orange')
plt.title('Loss')
plt.legend()
plt.show()

# Plot training and validation acc.  
plt.figure()
plt.plot(hist.history['accuracy'], label='Accuracy', color='teal')
plt.plot(hist.history['val_accuracy'], label='Val Accuracy', color='orange')
plt.title('Accuracy')
plt.legend()
plt.show()

# Evaluate model on test set
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

# For each batch in test dataset?
for batch in test:
    # Get X, y from batch
    X, y = batch
    # Predict yhat = model.predict(X)
    yhat = model.predict(X)
    # Update Precision with y, yhat
    pre.update_state(y, yhat)
    # Update Recall with y, yhat
    re.update_state(y, yhat)
    # Update Accuracy with y, yhat
    acc.update_state(y, yhat)

print('Precision:', pre.result().numpy())
print('Recall:', re.result().numpy())
print('Accuracy:', acc.result().numpy())

# Load single test image  
image_path = 'test\happy_8.jpg'
# Check if file exists at image_path
if not os.path.exists(image_path):
    # Raise FileNotFoundError
    raise FileNotFoundError(f"Image not found: {image_path}")

# Read image using cv2.imread
img = cv2.imread(image_path)

# Check if img is None
if img is None:
    # Raise FileNotFoundError
    raise ValueError("Failed to load image.")

# Image loaded successfully
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Preprocess (BGR→RGB, resize, norm)
resize = tf.image.resize(img, (256, 256))

# Predict image class (happy or sad)

# Display prediction result 
plt.imshow(resize.numpy().astype(int))
plt.title("Input Image")
plt.show()

# Predict image class (happy or sad)
yhat = model.predict(np.expand_dims(resize / 255, 0))

# if yhat > 0.5
if yhat > 0.5:
    # print sad
    print("Predicted class: Sad ")
else:
    # print happy
    print("Predicted class: Happy ")

# Save model (.h5)  
os.makedirs('models', exist_ok=True)
model.save(os.path.join('models', 'imageclassifier.h5'))
new_model = load_model(os.path.join('models', 'imageclassifier.h5'))

# Reload and test saved model   
new_prediction = new_model.predict(np.expand_dims(resize / 255, 0))
print("New model prediction:", new_prediction)
