import sys
sys.path.insert(0,'class')
import pathconfig
import preprocessing_images
import model

'''
   Script py to train the network CNN 
'''


paths = pathconfig.paths()                                  #instance to import paths
preprocessing = preprocessing_images.Preprocessing_Images() #instance to import preprocessing class
Model = model.Model()                                       #instance to import model class
model_cnn = model.ConvolutionNeuralNetwork()                #instance to import a  CNN model
path_model = model_cnn.path_model                           #path_model
path_weights = model_cnn.path_weights                       #path weights
print('Dataset for training set: {}'.format(preprocessing.trainingset ) )                #path of training set
print('Dataset for test set: {}'.format(preprocessing.testset))                          #path of test set

model = model_cnn.CNN()                                                #create a CNN model
history = model_cnn.training_model(model)                              #train the CNN model
name = model_cnn.name
Model.plot_history(history, name)                                      #plot the history of training
#Model.save_model(model,path_model,path_weights)                       #save the CNN model
