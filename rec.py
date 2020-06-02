# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import keras
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 5, activation = 'sigmoid'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Part 2 - Fitting the CNN to the images


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/Users/laxmanjeergal/Desktop/recognition/cow',
                                                 target_size = (64, 64),
                                                 batch_size = 32)
# test_set = test_datagen.flow_from_directory('dataset/test_set',
# target_size = (64, 64),
# batch_size = 32,
# class_mode = 'binary')
#fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1)

classifier.fit_generator(training_set,
                         steps_per_epoch = 5,
                         epochs = 15,
                         validation_data = None)

#indices
#training_set.class_indices

# Part 3 - Making new predictions

path ='/Users/laxmanjeergal/Desktop/cow31.jpg'

img = image.load_img(path)


x = image.img_to_array(img)



test_image = image.load_img(path, target_size = (64, 64))
test_image = image.img_to_array(test_image)
plt.imshow(test_image/255.)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(result)
print(training_set.class_indices)
if result[0][0] == 1:
    print("ram")
elif result[0][1]:
    print("cow1")
elif result[0][2]:
    print("cow2")
elif result[0][3]:
    print("cow3")
elif result[0][4]:
    print("cowsam")
