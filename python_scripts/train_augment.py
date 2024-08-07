import sys
import numpy as np
import pickle
import os
import random
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import (Conv2D, Dropout, GlobalAveragePooling2D,
                                    MaxPooling2D, Activation, Dense, Layer)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
from tensorflow.keras import backend as K 
import analysis, datasets

# Make GPU training deterministic
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']=str(0)

# Function to make models
def init_all_cnn_c(seed: int):
    
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(32, 32, 3), seed=0)) #input shape from keras cifar10 example
    model.add(Conv2D(96, (3, 3), input_shape=(32, 32, 3), padding='same', bias_regularizer=l2(1e-5),
                     kernel_regularizer=l2(1e-5), kernel_initializer=he_normal(seed), 
                     bias_initializer='zeros', activation='relu'))
    model.add(Conv2D(96, (3, 3), padding='same', kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5),
                     kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
    model.add(Conv2D(96, (3, 3), strides=2, padding='same', bias_regularizer=l2(1e-5),
                     kernel_regularizer=l2(1e-5), bias_initializer='zeros', activation='relu'))
    model.add(Dropout(0.5, seed=0))
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5),
                     kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5),
                     kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
    model.add(Conv2D(192, (3, 3), strides=2, padding='same',kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5),
                     kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
    model.add(Dropout(0.5, seed=0))
    model.add(Conv2D(192, (3, 3), padding='valid', kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5),
                     kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
    model.add(Conv2D(192, (1, 1), padding='valid', kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5), 
                     kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
    model.add(Conv2D(10, (1, 1), padding='valid', kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5), 
                     kernel_initializer=he_normal(seed), bias_initializer='zeros', activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9, clipnorm=500),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Scheduler callback
def scheduler(epoch, lr):
    if (epoch == 200 or epoch == 250 or epoch == 300):
        return lr * 0.1
    return lr

LR_Callback = LearningRateScheduler(scheduler)

# Trajectory callback
class Trajectory_Callback(Callback):
    '''
    Pre: Must define i, x_predict
    '''
    def on_epoch_end(self, epoch, logs=None):
        layer_arr = [7]
        if epoch in [0, 1, 2, 3, 4, 5,
                     6, 7, 8, 9,
                     49, 99, 149, 199, 249, 299, 349]:
            print('\n\nSnapshot weight', str(w), 'shuffle', str(s), 'at epoch', str(int(epoch)+1))
            acts = analysis.get_acts(self.model, layer_arr, x_predict, cocktail_blank=False)
            np.save('../outputs/representations/acts/ten_by_ten_3/w'+str(w)+'s'+str(s)+'e'+str(epoch)+'.npy', acts)
            print('\n')


# Cut off training if local minimum hit  
class Early_Abort_Callback(Callback):
    '''
    Pre: abort is set to False at the beginning of each training instance
    '''
    def on_epoch_end(self, epoch, logs=None):
        global abort
        if (epoch > 100 and logs.get('accuracy') <= 0.8):
            abort = True
            print('Acc:', logs.get('accuracy'))
            self.model.stop_training = True

ap = argparse.ArgumentParser()
ap.add_argument('-w', '--starting_weight', required=True)
args = vars(ap.parse_args())
w0 = int(args['starting_weight'])

for w in range(w0, w0+5):
    print('** Weight seed:', w)
    K.clear_session()
    for s in range(10):
        np.random.seed(0)
        tf.random.set_seed(0)
        random.seed(0)
        for r in range(100*w):
            random.randint(-10000, 10000)
        new_weight_seed = random.randint(-10000, 10000)
        print('new_weight_seed=', new_weight_seed)
    
        random.seed(0)
        for r in range(100*s):
            random.randint(-10000, 10000)
        new_shuffle_seed = random.randint(-10000, 10000)
        print('new_shuffle_seed=', new_shuffle_seed)
        
        trainData, testData = datasets.make_train_data(shuffle_seed=new_shuffle_seed, augment=True)
        x_predict, y_predict = datasets.make_predict_data(testData)
    
        model = init_all_cnn_c(seed=new_weight_seed)

        # Set flag to true if converges to local min
        abort = False
        history = model.fit(
            trainData,
            epochs=150,
            validation_data=testData.prefetch(tf.data.experimental.AUTOTUNE)\
                         .batch(128),
            callbacks=[LR_Callback, Early_Abort_Callback()])
        # Move onto the next shuffle candidate
    
        if not abort:
            save_model(model, '../outputs/models/ten-by-ten/w'+str(w)+'s'+str(s)+'.pb')
        else:
            raise ValueException('yo this hit local min')

