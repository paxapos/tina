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
from keras.models import Sequential,Model,load_model
from keras.optimizers import SGD
from keras.layers import BatchNormalization, Lambda, Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Reshape, Activation, Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image 
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

      # placeholder for input image
      input_image = Input(shape=(140,99,3))
      # ============================================= TOP BRANCH ===================================================
      # first top convolution layer
      top_conv1 = Convolution2D(filters=48,kernel_size=(11,11),strides=(4,4),
                                 input_shape=(281,199,3),activation='relu')(input_image)
      top_conv1 = BatchNormalization()(top_conv1)
      top_conv1 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_conv1)

      # second top convolution layer
      # split feature map by half
      top_top_conv2 = Lambda(lambda x : x[:,:,:,:24])(top_conv1)
      top_bot_conv2 = Lambda(lambda x : x[:,:,:,24:])(top_conv1)

      top_top_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv2)
      top_top_conv2 = BatchNormalization()(top_top_conv2)
      top_top_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_top_conv2)

      top_bot_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv2)
      top_bot_conv2 = BatchNormalization()(top_bot_conv2)
      top_bot_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_bot_conv2)

      # third top convolution layer
      # concat 2 feature map
      top_conv3 = Concatenate()([top_top_conv2,top_bot_conv2])
      top_conv3 = Convolution2D(filters=192,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_conv3)

      # fourth top convolution layer
      # split feature map by half
      top_top_conv4 = Lambda(lambda x : x[:,:,:,:96])(top_conv3)
      top_bot_conv4 = Lambda(lambda x : x[:,:,:,96:])(top_conv3)

      top_top_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv4)
      top_bot_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv4)

      # fifth top convolution layer
      top_top_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv4)
      top_top_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_top_conv5) 

      top_bot_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv4)
      top_bot_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_bot_conv5)

      # ============================================= TOP BOTTOM ===================================================
      # first bottom convolution layer
      bottom_conv1 = Convolution2D(filters=48,kernel_size=(11,11),strides=(4,4),
                                 input_shape=(224,224,3),activation='relu')(input_image)
      bottom_conv1 = BatchNormalization()(bottom_conv1)
      bottom_conv1 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_conv1)

      # second bottom convolution layer
      # split feature map by half
      bottom_top_conv2 = Lambda(lambda x : x[:,:,:,:24])(bottom_conv1)
      bottom_bot_conv2 = Lambda(lambda x : x[:,:,:,24:])(bottom_conv1)

      bottom_top_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv2)
      bottom_top_conv2 = BatchNormalization()(bottom_top_conv2)
      bottom_top_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_top_conv2)

      bottom_bot_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv2)
      bottom_bot_conv2 = BatchNormalization()(bottom_bot_conv2)
      bottom_bot_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_bot_conv2)

      # third bottom convolution layer
      # concat 2 feature map
      bottom_conv3 = Concatenate()([bottom_top_conv2,bottom_bot_conv2])
      bottom_conv3 = Convolution2D(filters=192,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_conv3)

      # fourth bottom convolution layer
      # split feature map by half
      bottom_top_conv4 = Lambda(lambda x : x[:,:,:,:96])(bottom_conv3)
      bottom_bot_conv4 = Lambda(lambda x : x[:,:,:,96:])(bottom_conv3)

      bottom_top_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv4)
      bottom_bot_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv4)

      # fifth bottom convolution layer
      bottom_top_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv4)
      bottom_top_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_top_conv5) 

      bottom_bot_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv4)
      bottom_bot_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_bot_conv5)

      # ======================================== CONCATENATE TOP AND BOTTOM BRANCH =================================
      conv_output = Concatenate()([top_top_conv5,top_bot_conv5,bottom_top_conv5,bottom_bot_conv5])

      # Flatten
      flatten = Flatten()(conv_output)

      # Fully-connected layer
      FC_1 = Dense(units=4096, activation='relu')(flatten)
      FC_1 = Dropout(0.6)(FC_1)
      FC_2 = Dense(units=4096, activation='relu')(FC_1)
      FC_2 = Dropout(0.6)(FC_2)
      output = Dense(units=10, activation='softmax')(FC_2)
      
      model = Model(inputs=input_image,outputs=output)
      #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
      sgd = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=True)
      model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
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
      image = np.array(img)
      hsv_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
      return Image.fromarray(hsv_image.astype(np.uint8))

   def train(self, productName: str):
      '''
      This function creates a model for each product, trains it based on
      the training and validation data and saves it in MODEL_PATH/productName
      Args:
         productName: a string with the name of the product with which the model will be trained and saved
      '''
      
      train_dir, validation_dir = self.__imageReader(productName)
      model = self.__createModel()
   
      filepath = (MODEL_PATH +"/"+ productName + ".h5")
      #checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
      callback = EarlyStopping(monitor='accuracy', baseline=0.05, patience=5, mode="max", verbose=1)
      callbacks_list = [callback]

      train_datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            preprocessing_function=self.__hsvFunction)

      validation_datagen = ImageDataGenerator(preprocessing_function=self.__hsvFunction)


      training_set = train_datagen.flow_from_directory(
                  train_dir,
                  target_size=(140, 99),
                  batch_size=10,
                  class_mode='categorical')

      validation_set = validation_datagen.flow_from_directory(
                  validation_dir,
                  target_size=(140, 99),
                  batch_size=10,
                  class_mode='categorical')


      history = model.fit(
            training_set,
            steps_per_epoch=4,
            epochs=EPOCHS_QUANTITY,
            validation_data=validation_set,
            validation_steps=1,
            callbacks= callbacks_list
            )
      model.summary()
      model.save(MODEL_PATH +"/"+ productName + ".h5")


      self.__accuracyGraph(history)
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

      model = tensorflow.keras.models.load_model(MODEL_PATH +"/"+ product +".h5")
      image = cv2.imread(img)
      hsv_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
      input_array = np.array(np.expand_dims(hsv_image, axis=0))
      array = model.predict(input_array)
      print(hsv_image)
      return array

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

   def __imageblur(self, imgpath):
      img = cv2.imread(imgpath)
      kernelsize = [64, 64]
      c = 0
      resultimage = cv2.blur(img, (kernelsize))
      while c < 15:
         resultimage = cv2.blur(resultimage, (kernelsize))
         c += 1
      return resultimage

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

   def __image_rgb(self, image):
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
      print(chans)
      r = feature_data[0]
      g = feature_data[1]
      b = feature_data[2]
      print(r, g, b)
      hsv = self.__rgb_to_hsv(r, g, b)
      return (hsv)
