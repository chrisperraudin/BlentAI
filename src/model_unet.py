import sys
import os

# Ajouter le dossier 'config' au chemin de recherche des modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))

# Importer la configuration
from config import cfg


import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from tensorflow.keras.utils import to_categorical
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#---------------------------------------------------------------------------
#UNET 2D
#---------------------------------------------------------------------------

def encoding_layer(input_layer,output_channels,kernel_size):
    conv1 = Conv2D(output_channels,(kernel_size, kernel_size),padding='same',kernel_initializer=tf.random_normal_initializer(0, 0.02)) (input_layer)
    conv1=relu(conv1)
    return conv1

def bottleneck(input_layer,output_channels,kernel_size):
    """
    Nous construisons le block bottleneck qui permet la liaison entre 
    l'encodeur et le décodeur. Il s'agit simplement de deux couches de convolutions 
    Conv2D suivie de la fonction d'activation Relu .
    """
    bottleneck1 = Conv2D(output_channels, (kernel_size, kernel_size), activation='relu', padding='same',kernel_initializer=tf.random_normal_initializer(0, 0.02)) (input_layer)
    bottleneck1 = relu(bottleneck1)
    bottleneck2 = Conv2D(output_channels, (kernel_size, kernel_size), activation='relu', padding='same',kernel_initializer=tf.random_normal_initializer(0, 0.02)) (bottleneck1)
    bottleneck2 = relu(bottleneck2)
    return bottleneck2 

def decoding_layer(input_layer,skip_layer,output_channels,kernel_size,stride):
    """
    Enfin, nous implémentons le décodeur qui regroupe une couche de déconvolution 
    Conv2DTranspose puis qui concatène l'entrée de la couche avec les skip connections 
    générés par l'encodeur. En sortie du bloc, on retrouve une couche de convolution 
    Conv2D avec la fonction Relu pour l'activation.
    """
    
    upconv1 = Conv2DTranspose(output_channels,  (kernel_size, kernel_size),strides=(stride,stride), padding='same',kernel_initializer=tf.random_normal_initializer(0, 0.02)) (input_layer)
    concat1 = concatenate([upconv1, skip_layer])
    conv1 = Conv2D(output_channels, kernel_size, activation='relu', padding='same',kernel_initializer=tf.random_normal_initializer(0, 0.02)) (concat1)
    conv1 = relu(conv1)
    return conv1

def build_model_UNET():
    inputs_coarse = Input((cfg['IMG_SIZE'],cfg['IMG_SIZE'],1))

    encoding_layer1=encoding_layer(inputs_coarse,64,3)
    pool1 = MaxPooling2D((2, 2),padding='same') (encoding_layer1)
    encoding_layer2=encoding_layer(pool1,128,3)
    pool2 = MaxPooling2D((2, 2),padding='same') (encoding_layer2)
    encoding_layer3=encoding_layer(pool2,256,3)
    pool3 = MaxPooling2D((1, 1),padding='same') (encoding_layer3)
    encoding_layer4=encoding_layer(pool3,512,3)
    pool4 = MaxPooling2D((1, 1),padding='same') (encoding_layer4)

    bottleneck1=bottleneck(pool4,1024,3)

    decoding_layer1= decoding_layer(bottleneck1, encoding_layer4,512,3,1)
    decoding_layer2= decoding_layer(decoding_layer1, encoding_layer3,256,3,1)
    decoding_layer3 = decoding_layer(decoding_layer2, encoding_layer2,128,3,2)
    decoding_layer4 = decoding_layer(decoding_layer3, encoding_layer1,64,3,2)

    outputs = Conv2D(3, (1, 1), activation='softmax') (decoding_layer4)

    model = Model(inputs=inputs_coarse, outputs=[outputs])
    optim=Adam(learning_rate=0.0001)
    model.compile(optimizer=optim, loss=['categorical_crossentropy'], metrics=['accuracy'])
    model.summary()
    
    return model


if __name__ == "__main__":
        # Construction du model
        model = build_model_UNET()
