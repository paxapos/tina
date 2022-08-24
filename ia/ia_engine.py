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
from tensorflow.keras import layers, regularizers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout
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
      model.add(Conv2D(32, (3, 3), input_shape=inputShape, padding = "same", activation='relu'))
      model.add(BatchNormalization())
      model.add(MaxPooling2D(2))
      model.add(Conv2D(32, (3, 3), padding = "same", activation='relu'))
      model.add(BatchNormalization())
      model.add(MaxPooling2D(2))
      model.add(Conv2D(32, (3, 3), padding = "same", activation='relu'))
      model.add(Conv2D(32, (3, 3), padding = "same", activation='relu'))
      model.add(Conv2D(32, (3, 3), padding = "same", activation='relu'))
      model.add(MaxPooling2D(2))

      model.add(Flatten())
      model.add(Dense(4096, activation='relu'))
      model.add(Dropout(0.5))
      model.add(Dense(4096, activation='relu'))
      model.add(Dropout(0.5))
      model.add(Dense(4096, activation='relu'))
      model.add(Dropout(0.5))
      model.add(Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(0.001)))

      model.compile(
         optimizer='adam',
         loss='categorical_crossentropy',
         metrics=["accuracy"]
      )
      return model


   def __imageSorter(self, path: str, productName: str):
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


      self.__accuracyGraph(history)
      self.visualize_conv_layer('conv_0')


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

   def __accuracyGraph (self, history):
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

   def __imageblur(self, imgpath, number, name):
      name = name
      img = cv2.imread(imgpath)
      kernelsize = [64, 64]
      c = 0
      resultimage = cv2.blur(img, (kernelsize))
      while c < 15:
         resultimage = cv2.blur(resultimage, (kernelsize))
         c += 1
      filename = "pics/Milanesas/new_validation/" + number + "/" + name
      print(filename)
      cv2.imwrite(filename, resultimage)

   def __rgb_to_hsv(self, r, g, b):
      r, g, b = r/255.0, g/255.0, b/255.0
      mx = max(r, g, b)
      mn = min(r, g, b)
      df = mx-mn
      if mx == mn:
         h = 0
      elif mx == r:
         h = (60 * ((g-b)/df) + 360) % 360
      elif mx == g:
         h = (60 * ((b-r)/df) + 120) % 360
      elif mx == b:
         h = (60 * ((r-g)/df) + 240) % 360
      if mx == 0:
         s = 0
      else:
         s = (df/mx)*100
      v = mx*100
      return h, s, v

   def __image_rgb(self):
      new_validation_dir = os.path.join(TRAINING_PICS_FOLDER, "Milanesas", "new_validation")
      a = os.listdir(new_validation_dir)
      for x in a:
         dirpath = new_validation_dir + "/" + x
         c = os.listdir(dirpath)
         number = x
         for image in c:
            imagepath = dirpath + "/" + image
            image = cv2.imread(imagepath)
            chans = cv2.split(image)
            colors = ('b', 'g', 'r')
            features = []
            feature_data = []
            counter = 0
            for (chan, color) in zip(chans, colors):
                    counter += 1
                    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
                    features.extend(hist)
                    elem = np.argmax(hist)

                    if counter == 1:
                        blue = int(elem)
                    elif counter == 2:
                        green = int(elem)
                    elif counter == 3:
                        red = int(elem)
                        feature_data = [red, green, blue]

            r = feature_data[0]
            g = feature_data[1]
            b = feature_data[2]
            print(r, g, b)
            hsv = self.rgb_to_hsv(r, g, b)
            print(hsv)
def visualize_conv_layer(layer_name):
  
  layer_output=model.get_layer(layer_name).output

  intermediate_model=tensorflow.keras.models.Model(inputs=model.input,outputs=layer_output)

  intermediate_prediction=intermediate_model.predict(input_array)
  
  row_size=4
  col_size=8
  
  img_index=0

  print(np.shape(intermediate_prediction))
  
  fig,ax=plt.subplots(row_size,col_size,figsize=(10,8))

  for row in range(0,row_size):
    for col in range(0,col_size):
      ax[row][col].imshow(intermediate_prediction[0, :, :, img_index])

      img_index=img_index+1

