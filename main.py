import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# # Load the MNIST dataset
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Normalize the data
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# # Define the model
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(x_train, y_train, epochs=3)

# # Save the model
# model.save('handwritten.keras')

# print("Model trained
model = tf.keras.models.load_model('handwritten.keras')

image_number = 1
while True:
    image_path = f"digits/digit{image_number}.png"
    if not os.path.isfile(image_path):
        print("No more images found in the directory.")
        break
    
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = np.invert(img)
        img = cv2.resize(img, (28, 28))  # Resize to 28x28 pixels
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        image_number += 1
