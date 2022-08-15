import tensorflow
import os 
import random
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
from pathlib import Path
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image as PImage
from tina.settings import BASE_DIR, TRAINING_PICS_FOLDER, MODEL_PATH, EPOCHS_QUANTITY, VALIDATION_PERCENTAGE


class IaEngine:
       
   def __imageReader (self, productName: str):
      '''
        Args:
          productName: name of the product folder to read
        Returns:
          train an validation folders of each product
      '''

      dirpath = BASE_DIR / TRAINING_PICS_FOLDER

      train_dir = os.path.join(dirpath, productName, 'train')
      validation_dir = os.path.join(dirpath, productName, 'validation')
      
      return train_dir, validation_dir
   
   def __createModel (self):
      '''
      This function creates the structure of the neural network
      and returns the model
      '''

      model = Sequential()
      inputShape = (281, 199, 3)

      model.add(Conv2D(32, (3, 3), padding="same",
	      input_shape=inputShape))
      model.add(Activation("relu"))
      model.add(Flatten())
      model.add(Dense(10))
      model.add(Activation("softmax"))

      model.compile(
         optimizer='adam',
         loss='categorical_crossentropy',
         metrics=["accuracy"]
      )
      return model


   def imageSorter__(self, path: str, productName: str):
      '''
      @TODO: write proper documentation for imageSorter__ function
      '''

      imgs = os.listdir(path)
      cantImgs = len(imgs)
      percentage = cantImgs * VALIDATION_PERCENTAGE / 100
      validationImages = []
      trainImages = imgs
      counter = 0

      while (counter < percentage):
         randomImage = random.choice(imgs)
         validationImages.append(randomImage)
         trainImages.remove(randomImage)
         shutil.move(path + "/" + randomImage, "training/pics/" + productName + "/validation")
         counter += 1
      for image in trainImages:
         shutil.move(path + "/" + image, "training/pics/" + productName + "/train")

   def train(self, productName: str):
      '''
      This function creates a model for each product, trains it based on
      the training and validation data and saves it in MODEL_PATH/productName
      Args:
         productName: a string with the name of the product with which the model will be trained and saved
      '''

      train_dir, validation_dir = self.__imageReader(productName)
      model = self.__createModel()
     
      train_datagen = ImageDataGenerator(rescale=1./255)
      val_datagen = ImageDataGenerator(rescale=1./255)

      train_generator = train_datagen.flow_from_directory(
      train_dir,
      target_size=(281, 199),
      batch_size=10,
      class_mode='categorical')

      validation_generator = val_datagen.flow_from_directory(
      validation_dir,
      target_size=(281, 199),
      batch_size=10,
      class_mode='categorical')

      print('Training...')
      history = model.fit(
      train_generator,
      steps_per_epoch=4,
      epochs=EPOCHS_QUANTITY,
      validation_data=validation_generator,
      validation_steps=1,
      verbose=2)
      print('Model Trained!')

      model.save(MODEL_PATH +"/"+ productName + ".h5")

   def predict(self, product: str, img: str):
      """
      this function takes an image and sends it to the neural network model corresponding to the product to return its score
         Args:
            product: type of product to predict 
            img: path to image to predict

         Returns:
            Numpy array(s) of predictions. Based on Keras Model.predict
      """

      model = tensorflow.keras.models.load_model(MODEL_PATH +"/"+ product +".h5")
      image = cv2.imread(img)
      input_array = np.array(np.expand_dims(image, axis=0))
      array = model.predict(input_array)
      return array

   def __accuracyGraph (self):
      '''
      This function generates a graphic of accuracy 
      for training and validation data 
      based on the amount of epochs 
      '''
      acc = history.history['acc']
      val_acc = history.history['val_acc']

      loss = history.history['loss']
      val_loss = history.history['val_loss']

      epochs = range(len(acc))

      plt.plot(epochs, acc)
      plt.plot(epochs, val_acc)
      plt.title('Training and validation accuracy')

      plt.figure()

      plt.plot(epochs, loss)
      plt.plot(epochs, val_loss)
      plt.title('Training and validation loss')

      plt.show()
