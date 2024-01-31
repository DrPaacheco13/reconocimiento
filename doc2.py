#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image.  In
#   particular, it shows how you can take a list of images from the command
#   line and display each on the screen with red boxes overlaid on each human
#   face.
#
#   The examples/faces folder contains some jpg images of people.  You can run
#   this program on them and see the detections by executing the
#   following command:
#       ./face_detector.py ../examples/faces/*.jpg
#
#   This face detector is made using the now classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image
#   pyramid, and sliding window detection scheme.  This type of object detector
#   is fairly general and capable of detecting many types of semi-rigid objects
#   in addition to human faces.  Therefore, if you are interested in making
#   your own object detectors then read the train_object_detector.py example
#   program.  
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

import sys
import os
import mysql
import dlib

detector = dlib.get_frontal_face_detector()
win = dlib.image_window()
# Conéctate a la base de datos
db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="reconocimiento"
)
cursor = db_connection.cursor()

# for f in sys.argv[1:]:
current_directory = os.getcwd()

# Concatena la ruta del script con el nombre de la imagen
image_path = os.path.join(current_directory, "elon2.jpg")


print("Processing file: {}".format(image_path))
# img_modi = cv2.imread('img/elon2.jpg')
# img_modi_rgb = cv2.cvtColor(img_modi, cv2.COLOR_BGR2RGB)
img = dlib.load_rgb_image(image_path)
# The 1 in the second argument indicates that we should upsample the image
# 1 time.  This will make everything bigger and allow us to detect more
# faces.
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
for i, d in enumerate(dets):
    left_coord, top_coord, right_coord, bottom_coord = d.left(), d.top(), d.right(), d.bottom()
    
    # Inserta en la base de datos
    insert_query = "INSERT INTO encodings (left_coord, top_coord, right_coord, bottom_coord) VALUES (%s, %s, %s, %s)"
    cursor.execute(insert_query, (left_coord, top_coord, right_coord, bottom_coord))
    db_connection.commit()

    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        i, d.left(), d.top(), d.right(), d.bottom()))
    
# Cierra la conexión a la base de datos
cursor.close()
db_connection.close()

win.clear_overlay()
win.set_image(img)
win.add_overlay(dets)
dlib.hit_enter_to_continue()

if (len(sys.argv[1:]) > 0):
    img = dlib.load_rgb_image(sys.argv[1])
    dets, scores, idx = detector.run(img, 1, -1)
    for i, d in enumerate(dets):
        print("Detection {}, score: {}, face_type:{}".format(
            d, scores[i], idx[i]))
else:
    print('no argv')