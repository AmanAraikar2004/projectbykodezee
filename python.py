
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the dataset
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
                                   
                                   
                                   

training_set = train_datagen.flow_from_directory('dataset/training_set',target_size = (64, 64),batch_size = 32,class_mode = 'categorical')
                                                 
                                                 
                                                 

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',target_size = (64, 64),batch_size = 32,class_mode = 'categorical')
                                            
                                            
                                            

# Build the model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(training_set, epochs=25, validation_data=test_set)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_set)
print('Test Accuracy: ', test_acc)

# Save the model for later use
model.save('waste_classification_model.h5')