import logging
from random import randint
from time import sleep
from tina.settings import CAMERA_MODULE

if CAMERA_MODULE:
    from picamera2 import Picamera2

logger = logging.getLogger()

def take_pictures(path, qty=1, delay=1):

    files = []

    picam2 = Picamera2()
    preview_config = picam2.still_configuration()
    picam2.configure(preview_config)
    picam2.start()
    try:
        for i in range(qty):
            directory = path
            pictureName = f'/image{randint(0, 999999999)}.jpg'
            data = picam2.capture_array()
            metadata = picam2.capture_file(str(directory) + pictureName)
            print(data)
            files.append(pictureName)
            sleep(delay)
    except Exception as e:
        print(e)
    finally:
        picam2.close()

    return files