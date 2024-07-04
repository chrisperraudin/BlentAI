import sys
import os

# Ajouter le dossier 'config' au chemin de recherche des modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))

# Importer
from config import cfg

import cv2
import numpy as np
import glob


def preprocessing(model, min_model, max_model):
    """Normalisation des images d'entraînement 
    entre 0 et 1 pour s'assurer que notre modèle converge facilement.
    """
    model = 2 * ((model - min_model) / (max_model - min_model)) - 1
    return model


def load_normalize_images(directory, startImg, endImg):
    """Charger les images pour l'entraînement"""
    train_img_list = sorted(glob.glob(directory + cfg['OUTPUT_DIR_IMAGE']  + '/*.png'))[startImg:endImg]

    train_img_list = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY) for img in train_img_list]

    max_coarse_models = np.max(np.array(train_img_list))
    min_coarse_models = np.min(np.array(train_img_list))

    train_img_list_norm = [preprocessing(image, min_coarse_models, max_coarse_models) for image in train_img_list]

    return train_img_list_norm


def rgb2mask(img_label):
    """transformer les valeurs des mask en valeurs labels"""
    image_rgb = img_label
    new_image = np.zeros((image_rgb.shape[0],image_rgb.shape[1],3)).astype('int')
    for row  in cfg['LABEL_TO_COLOR']:
        index = cfg['LABEL_TO_COLOR'][row]
        new_image[(image_rgb[:,:,0]==row[0])&
                  (image_rgb[:,:,1]==row[1])&
                  (image_rgb[:,:,2]==row[2])]=np.array([index,index,index]).reshape(1,3)
    new_image = new_image[:,:,0] 
    return(new_image)


def load_rgb2mask_labels(directory,startImg, endImg):
    
    train_label_list = sorted(glob.glob(directory + cfg['OUTPUT_DIR_MASK']+'/*.png'))[startImg:endImg]
                                                                                
    train_label_list = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in train_label_list]
    
    train_label_list = [rgb2mask(img[..., ::-1]) for img in train_label_list]
    
    return train_label_list  