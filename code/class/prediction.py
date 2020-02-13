import os
import csv
import cv2
import numpy as np
import pathconfig
from keras.preprocessing.image import ImageDataGenerator

paths = pathconfig.paths()

class Prediction:
    '''
        Class to prediction phase. it aims to exposes one method that save the images predicted from  dataset of test
        in corrispective class predicted.
    '''
    def __init__(self):
       self.path_images_test = paths.PATH_IMAGES_BLIND_TEST
    def prediction_save_images(self,path_pred_images:str, model:object):
        batch_size = 64

        pred_datagen = ImageDataGenerator(
            rescale = 1. / 255)

        pred_generator = pred_datagen.flow_from_directory(
            directory=self.path_images_test,
            target_size=(118, 224),
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False
        )

        val_steps=pred_generator.n//pred_generator.batch_size+1

        # predictions from transferNet
        preds = model.predict_generator(pred_generator,verbose=1,steps=val_steps)

        predictions = np.argmax(preds, axis=1)
        classes_dict_prediction = (pred_generator.class_indices)                      #vocabulary of classe with index
        classes_dict_prediction = dict((v,k) for k,v in classes_dict_prediction.items()) #vocabulary inverted
        predictions = [classes_dict_prediction[k] for k in predictions]               #all predictions
        n = 0
        for i in predictions:                                                         #for each class predicted
            image_ge = cv2.imread(self.path_images_test+pred_generator.filenames[n])   #read image from dataset of image
            name_image = pred_generator.filenames[n].replace('{}\\'.format(classes_dict_prediction[0]),'')\
                .replace('{}\\'.format(classes_dict_prediction[1]),'')\
                .replace('{}\\'.format(classes_dict_prediction[2]),'')\
                .replace('{}\\'.format(classes_dict_prediction[3]),'')    #name of original image
            path_save = path_pred_images+i                                     #path where save each image with class predicted
            cv2.imwrite(os.path.join(path_save, name_image), image_ge)         #write image in right folder (on class predicted) maintaining original name of file (image)
            n+=1
        print('images from path: %s have been predicted and saved in corrispective class: HAZE, RAINY,SNOWY and SUNNY (see the path: %s)' %(self.path_images_test,path_pred_images))

    def predictions_save_csv_file(self, path_pred_images: str, model: object):
        batch_size = 64
        pred_datagen = ImageDataGenerator(
            rescale=1. / 255)

        pred_generator = pred_datagen.flow_from_directory(
            directory=self.path_images_test,
            target_size=(118, 224),
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False
        )

        val_steps = pred_generator.n // pred_generator.batch_size + 1

        # predictions from transferNet
        preds = model.predict_generator(pred_generator, verbose=1, steps=val_steps)

        predictions = np.argmax(preds, axis=1)
        classes_dict_prediction = (pred_generator.class_indices)  # vocabulary of classe with index
        classes_dict_prediction = dict((v, k) for k, v in classes_dict_prediction.items())  # vocabulary inverted
        predictions = [classes_dict_prediction[k] for k in predictions]  # all predictions
        n = 0
        with open(paths.PATH_IMAGES_BLIND_TEST_PREDICTED_CSV, 'w', newline='') as file_csv:
            writer = csv.writer(file_csv)
            writer.writerow(["name image from blind test dataset", "weather class predicted"])
            for i in predictions:  # for each class predicted
                name_image = pred_generator.filenames[n].replace('{}\\'.format(classes_dict_prediction[0]), '') \
                    .replace('{}\\'.format(classes_dict_prediction[1]), '') \
                    .replace('{}\\'.format(classes_dict_prediction[2]), '') \
                    .replace('{}\\'.format(classes_dict_prediction[3]), '')  # name of original image
                writer.writerow([name_image, i])
                n += 1
        print(
            'images from path: %s have been predicted and saved in corrispective class: HAZE, RAINY,SNOWY and SUNNY (see the path: %s)' % (
            self.path_images_test, path_pred_images))