
import imp
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Data Preprocessing: Image Augmentation
# Training Set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

# Test Set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

# Building CNN
cnn = tf.keras.models.Sequential()
# Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))
# Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
# Convolution 2
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
# Flattening
cnn.add(tf.keras.layers.Flatten())
# Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
# Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(x=training_set, validation_data=test_set,epochs=4)

# Making Prediction
import numpy as np
from keras import utils
test_image = utils.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64,64))
test_image = utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image/255.0)
print(training_set.class_indices)
if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(prediction)