import tensorflow
import os 
import random
import shutil
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras.backend as K
from pathlib import Path
from tensorflow.keras import layers, regularizers, callbacks, preprocessing
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from keras.models import Sequential,Model,load_model
from keras.optimizers import SGD
from keras.layers import BatchNormalization, Lambda, Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Reshape, Activation, Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

# Tengo que ubicarme en el directorio padre para poder acceder a tina.settings
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
from tina.settings import BASE_DIR, TRAINING_PICS_FOLDER, MODEL_PATH, EPOCHS_QUANTITY, VALIDATION_PERCENTAGE



IMG_HEIGHT = 140
IMG_WIDTH = 99


class IaEngineBase:


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
   
   def __createModel (self, resultsQuantity: int):
      '''
      This function creates the structure of the neural network
      and returns the model
      '''
      sgd = tensorflow.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

      preTrainedModel = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
      mobilenetv2 = hub.KerasLayer(preTrainedModel, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
      mobilenetv2.trainable = False

      model = tensorflow.keras.Sequential([
         mobilenetv2,
         tensorflow.keras.layers.Dense(resultsQuantity, activation='softmax')
      ])
      model.compile(
         optimizer=sgd, 
         loss='categorical_crossentropy', 
         metrics=['accuracy']
      )
      return model


   def __visualize_conv_layer(self, layer_name):
      model = tensorflow.keras.models.load_model(MODEL_PATH +"/"+ "Milanesas" +".h5")
      layer_output=model.get_layer(layer_name).output
      
      intermediate_model=tensorflow.keras.models.Model(inputs=model.input,outputs=layer_output)
      image = cv2.imread("training/pics/Milanesas/train/0/pic_01.jpg")
      input_array = np.array(np.expand_dims(image, axis=0))
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
      plt.show()
          
   def __hsvFunction(self, img):
      #image = np.array(img)
      hsv_image = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
      return Image.fromarray(hsv_image.astype(np.uint8))

   def __accuracyGraph (self, history):
      '''
      This function generates a graphic of accuracy 
      for training and validation data 
      based on the amount of epochs 
      '''
      acc = history.history['accuracy']

      loss = history.history['loss']
      
      epochs = range(len(acc))

      plt.plot(epochs, acc)
      
      plt.title('Training and validation accuracy')

      plt.figure()

      plt.plot(epochs, loss)
      
      plt.title('Training and validation loss')

      plt.show()

class CookingIaEngine(IaEngineBase):


   def train(self, productName: str):
      '''
      This function creates a model for each product, trains it based on
      the training and validation data and saves it in MODEL_PATH/productName
      Args:
         productName: a string with the name of the product with which the model will be trained and saved
      '''
      
      train_dir, validation_dir = self._IaEngineBase__imageReader(productName)
      model = self._IaEngineBase__createModel(10)

      train_datagen = ImageDataGenerator( 
         rescale=1. / 255, 
         rotation_range=20, 
         horizontal_flip=True, 
         width_shift_range=0.2, 
         height_shift_range=0.2, 
         shear_range=0.2, 
         zoom_range=0.2
      )
      validation_datagen = ImageDataGenerator(rescale=1./255)

      training_set = train_datagen.flow_from_directory(
                  train_dir,
                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                  class_mode='categorical')

      validation_set = validation_datagen.flow_from_directory(
                  validation_dir,
                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                  class_mode='categorical')

      history = model.fit(
            training_set,
            steps_per_epoch=4,
            epochs=EPOCHS_QUANTITY,
            validation_data=validation_set,
            validation_steps=1,
            )
      model.summary()
      model.save(MODEL_PATH + "/" + productName + ".h5")


      self._IaEngineBase__accuracyGraph(history)
      print (model.metrics_names)
      print (model.evaluate(training_set, batch_size=10))
      #self.__visualize_conv_layer('asd')
      #plt.show()
   

   def predict(self, product: str, img: str):
      """
      this function takes an image and sends it to the neural network model corresponding to the product to return its score
         Args:
            product: type of product to predict 
            img: path to image to predict

         Returns:
            Numpy array(s) of predictions. Based on Keras Model.predict
      """

      loadimg = preprocessing.image.load_img( img, target_size=(IMG_HEIGHT, IMG_WIDTH) )
      npimg = preprocessing.image.img_to_array(loadimg, data_format=None, dtype=None)
      
      preTrainedModel = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
      model = load_model(MODEL_PATH + "/" + product + ".h5", custom_objects={'KerasLayer': hub.KerasLayer(preTrainedModel)})

      npimg = np.array(npimg).astype(float)/255
      npimg = cv2.resize(npimg, (IMG_HEIGHT, IMG_WIDTH))

      predict = model.predict( npimg.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 3) )
      return np.argmax(predict[0], axis=-1)


class CookingIaEngine(IaEngineBase):
   def __init__(self, lista_nombreProductos):
      self.lista_nombreProductos = lista_nombreProductos


   def train(self):
      '''
      This function creates a model for recognizing the products, trains it based 
      on the training and validation data and saves it in MODEL_PATH/foods
      '''
      
      train_dir, validation_dir = self._IaEngineBase__imageReader(productName)
      
      cantProductos = len(self.lista_nombreProductos)
      model = self._IaEngineBase__createModel(cantProductos)

      train_datagen = ImageDataGenerator( 
         rescale=1. / 255, 
         rotation_range=20, 
         horizontal_flip=True, 
         width_shift_range=0.2, 
         height_shift_range=0.2, 
         shear_range=0.2, 
         zoom_range=0.2
      )
      validation_datagen = ImageDataGenerator(rescale=1./255)

      training_set = train_datagen.flow_from_directory(
                  train_dir,
                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                  class_mode='categorical')

      validation_set = validation_datagen.flow_from_directory(
                  validation_dir,
                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                  class_mode='categorical')

      history = model.fit(
            training_set,
            steps_per_epoch=4,
            epochs=EPOCHS_QUANTITY,
            validation_data=validation_set,
            validation_steps=1,
            )
      model.summary()
      model.save(MODEL_PATH + "/foods.h5")


      self._IaEngineBase__accuracyGraph(history)
      print (model.metrics_names)
      print (model.evaluate(training_set, batch_size=10))
      #self.__visualize_conv_layer('asd')
      #plt.show()
   

   def predict(self, img: str):
      """
         this function takes an image and sends it to the neural network model to return its name
            Args:
               img: path to image to predict

            Returns:
               Name of the product.      
      """

      loadimg = preprocessing.image.load_img( img, target_size=(IMG_HEIGHT, IMG_WIDTH) )
      npimg = preprocessing.image.img_to_array(loadimg, data_format=None, dtype=None)
      
      preTrainedModel = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
      model = load_model(MODEL_PATH + "/foods.h5", custom_objects={'KerasLayer': hub.KerasLayer(preTrainedModel)})

      npimg = np.array(npimg).astype(float)/255
      npimg = cv2.resize(npimg, (IMG_HEIGHT, IMG_WIDTH))

      predict = model.predict( npimg.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 3) )
      return np.argmax(predict[0], axis=-1)
