
class paths(object):
    """ This is a class with goals to call all data paths from it. It  simplifies and streamlines the code from long paths.
    It is used following this rules:
    - in the file needs to include the file : import pathconfig,
    - create object from class : paths = pathconfig.paths()
    - call path from property of class: for example path_semcor = paths.TRAIN_DATASET
    Change all path in order to set own path and used them in the code.
    I remember that for path mappings the path are the same. So use this class to call them.
    Many files were deleted so
    """

    def __init__(self):
        #possible formats of files
        self.JSON = '.json'
        self.CSV = '.csv'
        self.MODEL = '.model'
        self.H5 = '.h5'


        #possible classes
        self.HAZE = '\\HAZE\\'
        self.RAINY = '\\RAINY\\'
        self.SNOWY = '\\SNOWY\\'
        self.SUNNY = '\\SUNNY\\'

        #Resources path base
        self.BASE_RESOURCES = _BASE_RES_PATH = '..\\resources\\'

        #Train Data Set
        self.PATH_TRAIN_DATASET_2000 = _BASE_RES_PATH + '\\MWI-public\\MWI-Dataset-1.1_2000\\'
        self.PATH_TEST_DATASET_400 = _BASE_RES_PATH + '\\MWI-public\\MWI-Dataset-1.1.1_400\\'
        self.PATH_TEST_DATASET_WEATHER = _BASE_RES_PATH + '\\TestSet_Weather\\Weather_Dataset'
        #Images Test
        self.PATH_IMAGES_BLIND_TEST = _BASE_RES_PATH + '1860363_images\\'


        #Models saved

        #CNN
        self.PATH_MODEL_CNN = _BASE_RES_PATH + 'model_cnn\\1860363_model{}'.format(self.H5)
        self.PATH_WEIGHTS_CNN = _BASE_RES_PATH + 'model_cnn\\1860363_weights{}'.format(self.H5)

        #TRANSFER LEARNING
        self.PATH_MODEL_TRA_LEARN = _BASE_RES_PATH + 'model_transfer_learning\\1860363_model{}'.format(self.H5)
        self.PATH_WEIGHTS_TRA_LEARN = _BASE_RES_PATH + 'model_transfer_learning\\1860363_weights{}'.format(self.H5)


        #Paths in which the predictions of images of test are saved

        #Predictiion obtained from CNN model
        self.PATH_PRED_CNN = _BASE_RES_PATH + '\\predictions_cnn\\'

        #Prediction obtained from CNN performed by tranfer learning VGG16
        self.PATH_PRED_TRANFER_LEARN = _BASE_RES_PATH + '\\predictions_transfer_learning\\'
        self.PATH_IMAGES_BLIND_TEST_PREDICTED_CSV = _BASE_RES_PATH + 'csv_image_classified{}'.format(self.CSV)
