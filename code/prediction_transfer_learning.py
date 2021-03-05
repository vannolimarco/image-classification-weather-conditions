import sys
sys.path.insert(0,'class')
import model
import pathconfig
import prediction


'''
   Script py to save prediction from dataset created to path for prediction about CNN model
'''

paths = pathconfig.paths()

Model = model.Model()
Prediction = prediction.Prediction()
model_Transfer_Learning= model.Transfer_Learning()         #Tranfer Learning model
path_images_test = Prediction.path_images_test
path_save_predicted_images = paths.PATH_IMAGES_BLIND_TEST_PREDICTED_CSV #path where the prediction from transfer Learning model are saved
path_model = model_Transfer_Learning.path_model            #the peth of model
path_weights = model_Transfer_Learning.path_weights        #the path of weights

model = Model.load_model(path_model,path_weights)          #load model

Prediction.predictions_save_csv_file(path_pred_images=path_save_predicted_images,model=model)     #save predictions in file csv
