import numpy as np
import sys
sys.path.insert(0,'class')
import model
import pathconfig
import preprocessing_images
import evaluation

'''
   Script py to evaluate the network Transfer Learning  
'''


paths = pathconfig.paths()


Model = model.Model()
Evaluation = evaluation.Evaluation()
model_Transfer_Learning= model.Transfer_Learning()         #Tranfer Learning model
path_test_set = paths.PATH_IMAGES_BLIND_TEST                     #the dataset chosen for evaluation. If you want change, you have to just change the
                                                           #path of dataset ( paths.PATH_IMAGES_TEST )
path_model = model_Transfer_Learning.path_model
path_weights = model_Transfer_Learning.path_weights

model = Model.load_model(path_model,path_weights)

Preprocessing_Images = preprocessing_images.Preprocessing_Images()
test_datagen, val_steps, classnames = Preprocessing_Images.get_set_from_path(setdata=path_test_set)

predictions = model.predict_generator(test_datagen,verbose=1,steps=val_steps)

Ypred = np.argmax(predictions, axis=1)  #label predicted
Ytest = test_datagen.classes            #label of test

Evaluation.plot_confusion_matrix_norm_multi_class(Ytest,Ypred)     #plot the confusion matrix
Evaluation.evaluation_errors(Ytest,Ypred,test_datagen,classnames)  #plot the errors about predictions
