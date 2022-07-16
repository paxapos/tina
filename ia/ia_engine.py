import tensorflow
import os 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tina.settings import BASE_DIR, TRAINED_PICS_FOLDER, MODEL_PATH

class IaEngine:
       
   def __imageReader (self, productName: str):
      '''
        Args:
          productName: name of the product folder to read

        Returns:
          train an validation folders of each product
      '''

      dirpath = str(BASE_DIR) + TRAINED_PICS_FOLDER


      # @TODO: read subfolder dirpath to get poroducts
      train_dir = os.path.join(dirpath, productName)

      validation_dir = os.path.join(dirpath, 'validation')
       
      train_raw_dir = os.path.join(train_dir, '0')

      train_burned_dir = os.path.join(train_dir, '8')

      validation_raw_dir = os.path.join(validation_dir, '0')

      validation_burned_dir = os.path.join(validation_dir, '8')

      train_raw_fnames = os.listdir(train_raw_dir)

      train_burned_fnames = os.listdir(train_burned_dir)

      return train_dir
      return validation_dir
    
   def __createModel (self):
      '''
      This function creates the structure of the neural network
      and returns the model
      '''

      # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
      # the three color channels: R, G, and B
      img_input = layers.Input(shape=(150, 150, 3))

      # First convolution extracts 16 filters that are 3x3
      # Convolution is followed by max-pooling layer with a 2x2 window
      x = layers.Conv2D(16, 3, activation='sigmoid')(img_input)
      x = layers.MaxPooling2D(2)(x)

      # Second convolution extracts 32 filters that are 3x3
      # Convolution is followed by max-pooling layer with a 2x2 window
      x = layers.Conv2D(32, 3, activation='sigmoid')(x)
      x = layers.MaxPooling2D(2)(x)
 
      # Third convolution extracts 64 filters that are 3x3
      # Convolution is followed by max-pooling layer with a 2x2 window
      x = layers.Conv2D(64, 3, activation='sigmoid')(x)
      x = layers.MaxPooling2D(2)(x)

      # Flatten feature map to a 1-dim tensor so we can add fully connected layers
      x = layers.Flatten()(x)

      # Create a fully connected layer with sigmoid activation and 512 hidden units
      x = layers.Dense(512, activation='sigmoid')(x)

      # Create output layer with a single node and sigmoid activation
      output = layers.Dense(1, activation='sigmoid')(x)

      # Create model:
      # input = input feature map
      # output = input feature map + stacked convolution/maxpooling layers + fully 
      # connected layer + sigmoid output layer
      model = Model(img_input, output)

      model.compile(loss='binary_crossentropy',
        optimizer=RMSprop(learning_rate=0.001),
        metrics=['acc'])
      return model 

   
   def __accuracyGraph (self):
      '''
      This function generates a graphic of accuracy 
      for training and validation data 
      based on the amount of epochs 
      '''

      # Retrieve a list of accuracy results on training and validation data
      # sets for each training epoch
      acc = history.history['acc']
      val_acc = history.history['val_acc']

      # Retrieve a list of list results on training and validation data
      # sets for each training epoch
      loss = history.history['loss']
      val_loss = history.history['val_loss']

      # Get number of epochs
      epochs = range(len(acc))

      # Plot training and validation accuracy per epoch
      plt.plot(epochs, acc)
      plt.plot(epochs, val_acc)
      plt.title('Training and validation accuracy')

      plt.figure()

     # Plot training and validation loss per epoch
      plt.plot(epochs, loss)
      plt.plot(epochs, val_loss)
      plt.title('Training and validation loss')

      plt.show()



   def train(self, productName: str):
      '''
      This function creates a model for each product, trains it based on
      the training and validation data and saves it in MODEL_PATH

      Args:
         productName: a string with the name of the product with which the model
         will be created
      '''

      train_dir = self.__imageReader(productName)
      validation_dir = self.__imageReader(productName)
      model = self.__createModel()
     
      # All images will be rescaled by 1./255
      train_datagen = ImageDataGenerator(rescale=1./255)
      val_datagen = ImageDataGenerator(rescale=1./255)

     # Flow training images in batches of 10 using train_datagen generator
      train_generator = train_datagen.flow_from_directory(
      train_dir,  # This is the source directory for training images
      target_size=(150, 150),  # All images will be resized to 150x150
      batch_size=10,
     # Since we use binary_crossentropy loss, we need binary labels
      class_mode='binary')

     # Flow validation images in batches of 10 using val_datagen generator
      validation_generator = val_datagen.flow_from_directory(
      validation_dir,
      target_size=(150, 150),
      batch_size=10,
      class_mode='binary')
      print('Training...')
      history = model.fit(
      train_generator,
      steps_per_epoch=4,  # 40 images = batch_size * steps
      epochs=100,
      validation_data=validation_generator,
      validation_steps=1,  # 10 images = batch_size * steps
      verbose=2)
      print('Model Trained!')
      model.save(MODEL_PATH)
    


   def predict(self, img: str):
      """
      this function takes an image and sends it to the neural network model to  
      return its score 
         Args:
            img: path to image to predict

         Returns:
            Numpy array(s) of predictions. Based on Keras Model.predict
      """
      model = tensorflow.keras.models.load_model(MODEL_PATH)
      model.summary()
      img = load_img(img) 
      img_input = img_to_array(img.resize((150, 150)))
      print (img_input.shape)
      narr = model.predict(img_input)

      score = narr[0]
      return score
