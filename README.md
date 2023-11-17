# tina
Cook bot API. Use IA for cooking. Take pictures with your phone so you can know if product is done


# Documentación
Aquí incorporaremos toda la documentación necesaria para unir las APPS, la API y conectar con el robot cocinero




# First steps

git clone git@github.com:paxapos/tina.git

pip install -r requirements.txt 

copy /tina/settings.py.copy to /tina/settings.py and change SECRET_KEY 

test that is correct working
python manage.py runserver



# Imagenes de entrenamienmto

https://github.com/paxapos/tina/files/9229521/Milanesas.zip


# PASANTIAS-2023

Avances: 
    Logramos que TINA diferenciara entre los niveles de cocción y las comidas.

    Para probarlo se puede acceder a ia/ñoquisGenerator.py y colocar las imagenes resultantes en training/pics. Luego pueden ejecutar el archivo ia/ia_engine.py agregando las siguientes lineas de código:

        productsIA = ProductIaEngine(['ñoquis']) # Le mando como parametro la lista de los nombres ordenada segun la carpeta de entrenamiento para mostrar la comida resultante.

        IA = CookingIaEngine()

        IA.train('ñoquis')
        productsIA.train('foods')


        def predictImages(imageURL):
            foodType = productsIA.predict(imageURL)
            foodCookingLevel = IA.predict(foodType, imageURL) 
            print('Food:', foodType)
            print('Cooking level:', foodCookingLevel)

        dirpath = BASE_DIR / TRAINING_PICS_FOLDER

        imageURL = os.path.join(dirpath, 'ñoquis/validation/7/7_imagen_1.png')
        predictImages(imageURL)

Errores: 
    Tuvimos problemas con las diversas versiones en las que se había desarrollado hasta el momento. Solucionamos esto adaptando el código a las versiones actuales. También hubo un contratiempo con la importación de MEDIA_ROOT. Además, enfrentamos problemas al principio durante la instalación de algunas extensiones, los cuales fuimos solucionando a la larga.

Nuevos objetivos:
    A futuro, el objetivo es hacer la interacción del back con el front y re-entrenar los modelos con imágenes de comida reales (pensamos en sacarlas por cuenta propia o generarlas a traves de alguna IA)