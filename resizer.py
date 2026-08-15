import os
import cv2
import numpy as np
resized_pixels = []


#Resizing the daisy folder images
for i in os.listdir("flowers/daisy"):
    file_path = os.path.join("flowers/daisy",i)
    read_file = cv2.imread(file_path)
    resized_image = cv2.resize(read_file,(32,32))
    resized_pixels.append(resized_image)
    

#Resizing the dandelion folder images
for i in os.listdir("flowers/dandelion"):
    file_path_dandelion = os.path.join("flowers/dandelion",i)
    read_file_dandelion = cv2.imread(file_path_dandelion)
    resized_image_dandelion = cv2.resize(read_file,(32,32))
    resized_pixels.append(resized_image_dandelion)
    

#Resizing the rose folder images
for i in os.listdir("flowers/rose"):
    file_path_rose = os.path.join("flowers/rose",i)
    read_file_rose = cv2.imread(file_path_rose)
    resized_image_rose = cv2.resize(read_file,(32,32))
    resized_pixels.append(resized_image)
    


#Resizing the sunflower folder images
for i in os.listdir("flowers/sunflower"):
    file_path_sunflower = os.path.join("flowers/sunflower",i)
    read_file_sunflower = cv2.imread(file_path_sunflower)
    resized_image_sunflower = cv2.resize(read_file,(32,32))
    resized_pixels.append(resized_image)
    


#Resizing the tulip folder images
for i in os.listdir("flowers/tulip"):
    file_path_tulip = os.path.join("flowers/tulip",i)
    read_file_tulip = cv2.imread(file_path_sunflower)
    resized_image_tulip = cv2.resize(read_file,(32,32))
    resized_pixels.append(resized_image)

np.save("resised_pixels_final",resized_pixels)