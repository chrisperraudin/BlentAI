
cfg = dict()



#Les chemins des fichiers sources LOCAUX
cfg['DATASET_PATH']         = 'D:/Datasets/brain_tumor_2019/MICCAI_BraTS_2019_Data_Training/'
cfg['TRAIN_DIR']            = 'D:/Datasets/brain_tumor_2019/Detail/TRAIN/'
cfg['VALID_DIR']            = 'D:/Datasets/brain_tumor_2019/Detail/VALID/'
cfg['TESTS_DIR']            = 'D:/Datasets/brain_tumor_2019/Detail/TESTS/'

#Les chemins des fichiers sources SAGEMAKER
# cfg['DATASET_PATH']         = '/home/ec2-user/SageMaker/brain_tumor_2019/MICCAI_BraTS_2019_Data_Training/'
# cfg['TRAIN_DIR']            = '/home/ec2-user/SageMaker/brain_tumor_2019//Detail/TRAIN/'
# cfg['VALID_DIR']            = '/home/ec2-user/SageMaker/brain_tumor_2019/Detail/VALID/'
# cfg['TESTS_DIR']            = '/home/ec2-user/SageMaker/brain_tumor_2019//Detail/TESTS/'

cfg['DATASET_PATH_SUBDIRS'] = ['LGG/', 'HGG/']

#Les chemins des images générées
cfg['OUTPUT_DIR_NII']       = 'Nii/'
cfg['OUTPUT_DIR_IMAGE']     = 'Images/'
cfg['OUTPUT_DIR_LABELS']    = 'Labels/'
cfg['OUTPUT_DIR_MASK']      = 'Mask/'


#DATA GENERATOR
cfg['VOLUME_SLICES']        = 144    #Nombre de segments de l'image .NII à générer (144)
cfg['IMG_SIZE']             = 240   #Taille des images à générer (60)
cfg['NBIMGNII_TO_GENERATE'] = None    #Nbre d'images NII à traiter. SI valeur = None on traite tout (None = ok)
cfg['SPLIT_TRAIN']          = 70    #Pourcentage d'images NII à utiliser pour le train
cfg['SPLIT_VALID']          = 20    #Pourcentage d'images NII à utiliser pour la validation
cfg['SPLIT_TESTS']          = 10    #Pourcentage d'images NII à garder pour les tests


#TRAIN BLENT
#cfg['UNET_EPOCHS']          = 5    #Nombre d'épochs sur le train UNET (15)  (2=OK)
#cfg['UNET_BATCH_SIZE']      = 16    #Taille du batch UNET pour le train (96)  (8=OK)
#cfg['NB_IMG_BY_TRAIN_FIT']  = 500   #Dans le FIt, nbre d'image par boucle pour otpimiser mem GPU (500)  (50=OK)

#TRAIN LOCAL
cfg['UNET_EPOCHS']          = 5    #Nombre d'épochs sur le train UNET (15)  (2=OK)
cfg['UNET_BATCH_SIZE']      = 8    #Taille du batch UNET pour le train (96)  (8=OK)
cfg['NB_IMG_BY_TRAIN_FIT']  = 50   #Dans le FIt, nbre d'image par boucle pour otpimiser mem GPU (500)  (50=OK)

cfg['MODEL_NAME']           = 'Model_240px_all_nii_epo_bs16_trainfit500_Blent.h5' #nO bs08 ko - bs04
                             

cfg['LABEL_TO_COLOR']       = {
                               (0, 0,0) : 0,        #background  black
                               (255, 255,255) : 1,  #brain       white
                               (0, 0,255) : 2,      #tumor        red
                              } 

#TEST/VALIDATION
cfg['VALID_OR_TRAIN_MODE']  = 'VALID' #Valeurs possible 'VALID' ou 'TESTS' pour choix du jeux de données à passer au modèle
cfg['NBIMG_TO_CHECK']       = 100      #Nbre d'images du répertoire VALIDATION/TESTS à passer au modèle. SI valeur = None, on traite tout
cfg['DISPLAY_GRAPH']        = False  #Affiche le graphique de comparaison TRue/False
