import sys
sys.path.insert(0,'class')
import pathconfig
import matplotlib.pyplot as plt
import preprocessing_images
import model


'''
   Script py to train the Transfer learning network 
'''

paths = pathconfig.paths()                                  #instance to import paths
preprocessing = preprocessing_images.Preprocessing_Images() #instance to import preprocessing class
Model = model.Model()                                       #instance to import model class
model_Tran = model.Transfer_Learning()                      #instance to import a  CNN model
path_model = model_Tran.path_model                          #path_model

path_weights = model_Tran.path_weights                       #path weights
print('Dataset for training set: {}'.format(preprocessing.trainingset ) )                #path of training set
print('Dataset for test set: {}'.format(preprocessing.testset))                          #path of test set


history = model_Tran.training_model()                              #train the CNN model

name = model_Tran.name
Model.plot_history(history, name)                                  #plot the history of training
#Model.save_model(model,path_model,path_weights)                   #save model
