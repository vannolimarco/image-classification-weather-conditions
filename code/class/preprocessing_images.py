import pathconfig
from keras.preprocessing.image import ImageDataGenerator

paths = pathconfig.paths()

class Preprocessing_Images:
    '''
         Preprocessing Class for image preprocessing phase. It implement a several number of methods used to
         perform some action for preprocessing. It is the main class with which is possible
         call all methods destined to perform preprocessing for image augumentation.
    '''
    def __init__(self):
        self.trainingset = paths.PATH_TRAIN_DATASET_2000  #the dataset for training
        self.testset = paths.PATH_TEST_DATASET_400        #the dataset for set of validation (test)
        self.testset_weather_dataset = paths.PATH_TEST_DATASET_WEATHER   #the dataset for testset chosen as second choice as testset
        self.batch = 64                                   #value of batch
    def get_train_test_set(self):
        '''
           this method return the training/testing generator, the step-per-epochs and val-step. The samples
           taken for training are generated from dataset MWI-Dataset-1.1.1_2000 and 400 as training and test set.
        '''
        train_datagen = ImageDataGenerator(
            rescale=1. / 255, \
            zoom_range=0.1, \
            rotation_range=10, \
            width_shift_range=0.1, \
            height_shift_range=0.1, \
            horizontal_flip=True, \
            vertical_flip=False)

        train_generator = train_datagen.flow_from_directory(
            directory=self.trainingset,
            target_size=(118, 224),
            color_mode="rgb",
            batch_size=self.batch,
            class_mode="categorical",
            shuffle=True
        )

        test_datagen = ImageDataGenerator(
            rescale=1. / 255)

        test_generator = test_datagen.flow_from_directory(
            directory=self.testset,
            target_size=(118, 224),
            color_mode="rgb",
            batch_size=self.batch,
            class_mode="categorical",
            shuffle=False
        )

        num_samples = train_generator.n           #number of samples
        num_classes = train_generator.num_classes #number of classes
        input_shape = train_generator.image_shape #shape of image

        classnames = [k for k, v in train_generator.class_indices.items()]

        print("Image input %s" % str(input_shape))
        print("Classes: %r" % classnames)

        print('Loaded %d training samples from %d classes.' % (num_samples, num_classes))
        print('Loaded %d test samples from %d classes.' % (test_generator.n, test_generator.num_classes))

        steps_per_epoch = train_generator.n // train_generator.batch_size
        val_steps = test_generator.n // test_generator.batch_size + 1

        return train_generator,test_generator,steps_per_epoch,val_steps,input_shape

    def get_set_from_path(self, setdata:str):
        '''
           this method return the training/testing generator and val-step but from one dataset
           specificated by parameter of the method. The samples
           taken from dataset chosen as parameter. This method takes as parameter the dataset chosen.
        '''
        test_datagen = ImageDataGenerator(
            rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
            directory=setdata,
            target_size=(118, 224),
            color_mode="rgb",
            batch_size=self.batch,
            class_mode="categorical",
            shuffle=False
        )
        classnames = [k for k, v in test_generator.class_indices.items()] #the classnames
        val_steps = test_generator.n // test_generator.batch_size + 1     #the val steps
        return test_generator,val_steps,classnames