import sys
sys.path.insert(0,'class')
import model
import pathconfig
import prediction


'''
   Script py to save prediction from dataset created byt myself to path for predction about CNN model
'''

paths = pathconfig.paths()

Model = model.Model()
Prediction = prediction.Prediction()
model_cnn = model.ConvolutionNeuralNetwork()     #Convolution neural network
path_images_test = Prediction.path_images_test
path_save_predicted_images = paths.PATH_PRED_CNN #path where the prediction from CNN model are saved
path_model = model_cnn.path_model                #the peth of model
path_weights = model_cnn.path_weights            #the path of weights

model = Model.load_model(path_model,path_weights)

Prediction.prediction_save_images(path_pred_images=path_save_predicted_images,model=model)   #save image in specific folder predictions
