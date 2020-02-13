import keras
import matplotlib.pyplot as plt
import numpy as np
import pathconfig
import preprocessing_images
from keras import applications
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, MaxPooling2D, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Input
from keras.models import Sequential, Model
from keras.models import model_from_json

paths = pathconfig.paths()

class Model:
    '''
         Class to define the two  models implemented: The convolution neural network and The Convolution neural network
         performed by VGG16 (Transfer Learning). This class is based two class, each of them call a expons a specific methods
         to build and perform or a classic convolution neural network (CNN) or a convolution neural network with Ttranfer Learning
         (VGG16).
    '''
    def save_model(self,model: object, path_model: str, path_weights: str):
        '''
            this method aims to save the model given the model and the paths where there are the model and weights.
        '''
        model_json = model.to_json()
        with open(path_model, "w") as json_file:
            json_file.write(model_json)
            print("Saved model to file (format json): " + path_model)
        # serialize weights to HDF5
        model.save_weights(path_weights)
        print("Saved weights of model to file : " + path_weights)

    def load_model(self,path_model_saved: str, path_weights_saved: str):
        '''
            this method aims to load the model and weiths already saved in teh paths indicated as parameters.
        '''
        json_file = open(path_model_saved, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_load = model_from_json(loaded_model_json)
        model_load.load_weights(path_weights_saved)
        print("Loaded model from path: %s" % (path_weights_saved))
        return model_load

    def plot_history(self,history, name):
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title(name + ' accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(name + ' loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

class ConvolutionNeuralNetwork:
    '''
       this is a class that aims to build and train a convolution neural network.
    '''
    def __init__(self):
        self.path_model = paths.PATH_MODEL_CNN         #the path where the model for this network has to be saved
        self.path_weights = paths.PATH_WEIGHTS_CNN     #the path where the weights for this network have to be saved
        self.name = 'convolution neural network'
        self.relu = 'relu'                             #attivaction funciton 'relu tag
        self.softmax = 'softmax'                       #attivaction function 'softmax' tag
        self.categorical_crossentropy = 'categorical_crossentropy'  #loss evaluation categorical crossentrapy tag
        self.adam = 'adam'                             #optimizer adam
        self.acc = 'acc'                               #accuracy tag
        self.loss = 'loss'                             #loss tag
        self.val_loss = 'val_loss'                     #val loss tag
        self.val_acc = 'val_acc'                       #val accuracy tag
        self.input_shape = (118, 224, 3)               #input shape
        self.size_kernel = (3,3)                       #size kernel
        self.epochs = 100                              #epochs
    def CNN(self):
        '''
            this method aims to build a Convolution neural netowrk with the following layers:
            - two conv2D layers
            - two MaxPooling2D layers
            - four Dropout layers
            - three Dense layers
        '''
        model = Sequential()

        # add a Convolution layer with dropout and MaxPooling process
        model.add(Conv2D(filters=16, kernel_size=self.size_kernel, activation=self.relu, input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=self.size_kernel))
        model.add(Dropout(0.5))

        # add a Convolution layer with dropout and MaxPooling process
        model.add(Conv2D(filters=64, kernel_size=self.size_kernel, activation=self.relu))
        model.add(MaxPooling2D(pool_size=self.size_kernel))
        model.add(Dropout(0.25))

        # flat the output of a Conv layer
        model.add(Flatten())

        # add a Dense Layer
        model.add(Dense(128, activation=self.relu))
        model.add(Dropout(0.2))

        # add a Dense Layer
        model.add(Dense(64, activation=self.relu))
        model.add(Dropout(0.2))

        # add a final Dense Layer for four class: haze,rainy,snowy and sunny
        model.add(Dense(4, activation=self.softmax))

        model.compile(loss=self.categorical_crossentropy, optimizer=self.adam, metrics=[self.acc])
        model.summary()
        return model

    def training_model(self,model):
        '''
            this method aims to train the model CNN compiled
        '''
        Preprocessing_Images = preprocessing_images.Preprocessing_Images()
        train_generator,test_generator,steps_per_epoch,val_steps,input_shape =  Preprocessing_Images.get_train_test_set()
        checkpoint = ModelCheckpoint(self.path_weights,
                                     monitor=self.val_acc,
                                     mode='min',
                                     save_best_only=False,
                                     verbose=1)
        try:
            history = model.fit_generator(train_generator, epochs=self.epochs, verbose=1, callbacks=[checkpoint], \
                                                   steps_per_epoch=steps_per_epoch, \
                                                   validation_data=test_generator, \
                                                   validation_steps=val_steps)
        except KeyboardInterrupt:
            pass

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, self.epochs), history.history[self.loss], label="train_loss")
        plt.plot(np.arange(0, self.epochs), history.history[self.val_loss], label="val_loss")
        plt.plot(np.arange(0, self.epochs), history.history[self.acc], label="train_acc")
        plt.plot(np.arange(0, self.epochs), history.history[self.val_acc], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch N°")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        return history

class Transfer_Learning:
    '''
        This is a class that aims to build and train a convolution neural network performed with a pre-trained model
        VGG16, so a transfer learning process.
    '''
    def __init__(self):
        self.path_model = paths.PATH_MODEL_TRA_LEARN
        self.path_weights = paths.PATH_WEIGHTS_TRA_LEARN
        self.name = 'transfer learning network (VGG16)'
        self.imagenet = 'imagenet'     #the dataset used to train the VGG16 model
        self.relu = 'relu'  # attivaction funciton 'relu tag
        self.softmax = 'softmax'  # attivaction function 'softmax' tag
        self.categorical_crossentropy = 'categorical_crossentropy'  # loss evaluation categorical crossentrapy tag
        self.adam = 'adam'  # optimizer adam
        self.acc = 'acc'  # accuracy tag
        self.loss = 'loss'  # loss tag
        self.val_loss = 'val_loss'  # val loss tag
        self.val_acc = 'val_acc'  # val accuracy tag
        self.size_kernel = (3, 3)  # size kernel
        self.epochs = 100  # epochs
        self.transferNet = 'transferNet'
        self.name_output_extractor = "block5_pool" # choose the layer from which you can get the features
                                                   # (block5_pool) layer
        self.trainable_layers = ["block5_conv3"]   # (block5_conv3) layer

    def load_VGG16(self,input_shape):
        '''
            this method aims to load the VGG16 network pre-trained with dataset 'imagenet'
        '''
        # define input tensor
        input = Input(shape=input_shape)

        # load a pretrained model on imagenet without the final dense layer
        feature = applications.vgg16.VGG16(include_top=False, weights=self.imagenet, input_tensor=input)

        model = Model(input=input, output=feature.output)

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=self.adam, metrics=[self.acc])

        return model

    def TRANSFER_LEARNING_NET(self,feature_extractor, num_classes, output_layer_name, trainable_layers):
        '''
            this method aims to build a netowork using Tranfer Learning process and VGG16 network
        '''
        # get the original input layer tensor
        input_transfer = feature_extractor.get_layer(index=0).input

        # set the feature extractor layers as non-trainable
        for idx, layer in enumerate(feature_extractor.layers):
            if layer.name in trainable_layers:
                layer.trainable = True
            else:
                layer.trainable = False

        # get the output tensor from a layer of the feature extractor
        output_extractor = feature_extractor.get_layer(name=output_layer_name).output

        # flat the output of a Conv layer
        flatten = Flatten()(output_extractor)
        flatten_norm = BatchNormalization()(flatten)

        # add a Dense layer
        dense = Dropout(0.4)(flatten_norm)
        dense = Dense(200, activation=self.relu)(dense)
        dense = BatchNormalization()(dense)

        # add a Dense layer
        dense = Dropout(0.4)(dense)
        dense = Dense(100, activation=self.relu)(dense)
        dense = BatchNormalization()(dense)

        # add the final output layer (four class: haze,rainy,snowy and sunny)
        dense = BatchNormalization()(dense)
        dense_final_layer = Dense(num_classes, activation=self.softmax)(dense)

        model = Model(input=input_transfer, output=dense_final_layer, name=self.transferNet)

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=self.adam, metrics=[self.acc])
        model.summary()
        return model

    def training_model(self):
        '''
            this method aims to train the Tranfer Leanring model
        '''
        Preprocessing_Images = preprocessing_images.Preprocessing_Images()
        train_generator, test_generator, steps_per_epoch, val_steps, input_shape = Preprocessing_Images.get_train_test_set()
        feature_extractor = self.load_VGG16(input_shape)
        feature_extractor.summary()

        # build the transfer model
        transfer_model = self.TRANSFER_LEARNING_NET(feature_extractor, train_generator.num_classes, self.name_output_extractor, self.trainable_layers)
        transfer_model.summary()

        # fit the transferNet on the training data
        steps_per_epoch = train_generator.n // train_generator.batch_size
        val_steps = test_generator.n // test_generator.batch_size + 1

        stopping = checkpoint = ModelCheckpoint(self.path_weights,
                                                monitor=self.val_acc,
                                                mode='min',
                                                save_best_only=False,
                                                verbose=1)

        try:
            history = transfer_model.fit_generator(train_generator, epochs=self.epochs, verbose=1, callbacks=[stopping], \
                                                   steps_per_epoch=steps_per_epoch, \
                                                   validation_data=test_generator, \
                                                   validation_steps=val_steps)
        except KeyboardInterrupt:
            pass

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, self.epochs), history.history[self.loss], label="train_loss")
        plt.plot(np.arange(0, self.epochs), history.history[self.val_loss], label="val_loss")
        plt.plot(np.arange(0, self.epochs), history.history[self.acc], label="train_acc")
        plt.plot(np.arange(0, self.epochs), history.history[self.val_acc], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch N°")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        return history

