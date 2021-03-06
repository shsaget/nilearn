"""
RSA MODULE : Module for Representational Similarity Analysis (Kriegeskorte 2008)

!!!!!    CHECK THE DATA_STRUCTURE_RECOMMANDATION FILE    !!!!!
!!!!!  TO SEE HOW THE DATA ARE EXPECTED TO BE STRUCURED  !!!!!
!!!!!       FOR THE GOOD BEHAVIOUR OF THIS MODULE        !!!!!
!!!!!  YOU CAN USE structure_info() FUNCTION TO SEE IT   !!!!!  #### A faire ####


This approach consists in observing the similarity between various
conditions in their representation's space.
Overall this approach alows to compare these simili represeantions between
various models (like the primate brains, computationnal models, behavioural models...)

Functions in this module are specially implemented for using neuro-imaging fMRI data.
Their inputs are the extracted signal values from brain images.
Please see ****_values_extraction.py


Prefixe signification :
NU = Not Used / Usefull (maybe contain errors)
NW = Not Written
"""
# Authors : Shinji Saget (sh.saget.pro@gmail.com)
#
#############################################################################################################################
## IMPORTS ####################################################################################################### IMPORTS ##
#############################################################################################################################
import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats.mstats import spearmanr
from scipy.spatial.distance import euclidean
from scipy.sparse import save_npz, load_npz

## Time and progress
import time
from tqdm import tqdm

## Parallelization
from joblib import Parallel, delayed

## Self_import

from nilearn import image, masking, decoding

#############################################################################################################################
## Utilities ################################################################################################### Utilities ##
#############################################################################################################################

def BetaNamesConstruction(betas_path, verbose = 1):
    """
    This function allows to construct variables containing various information about the data
    that will be use in the RSA

    Parameters :
        - betas_path : path containing the subjectes folder                     |  type str
                       with the betas images within each one
        - verbose : control the verbosity of the function                      |  type int

    Outputs :
        - subs : list of the subject prensent in the 'betas' folder             |  type list
        - nsubs : number of subject                                             |  type int
        - betas : dictionnary containing key for each subject                   |  type dict
                  Each key leading to the list of the subject's betas
    """
    subs = os.listdir(betas_path)
    subs.sort()
    nsubs = len(subs)
    betas = {}

    # on parcours les dossier des différents sujets pour en extraire tous les betas et les ranger dans 1 dico
    for i in np.arange(nsubs):
        subject_path = betas_path + '/' + subs[i]  # on prend le chemin du dossier du sujet en cours
        betas[i] = os.listdir(subject_path) # on y sélectionne tout les betas_path
        betas[i].sort()
    if verbose > 0:
        print('Nombre de sujets : {0}' .format(nsubs))
        print("Nombre d'images par sujet: {0}" .format(len(betas[0])))

    return(subs, nsubs, betas)

def GucluLabels(nsubs, mode = 'detailled'):
    """
        Fonction permettant de créer la liste des labels correspondant au nombre
        de sujets présents dans l'étude de GUCLU basée sur les données StudyForrest

        Les différents modes disponibles sont :
        - 'detailled' : permet d'avoir les labels exacts des stimuli (5 labels différents pour un même genres)
        - 'by_genre'  : permet d'avoir les labels définis uniquement sur les genres (5 fois moins de labels)
    """

    if (mode != 'detailled' and mode != 'by_genre'):
        print('Mode choisi incorrect. Le mode par défaut "detailled" est activé ')
        mode = 'detailled'

    #  Pour chaque sujet on a 25 betas rangés par ordre alphabétique
    labels = []  # list contenant tout les stimuli de chaque run de chaque sujet (25*8*12 = 2400)
    stim_in_run = []   # liste des stimuli contenu dans une run (25*1)
    # création d'une liste de stimuli pour une session :
    genres = ['ambiant', 'country', 'metal', 'rocknroll', 'symphonic']

    ## création des labals d'une run
    for g in np.arange(5):
        if (mode == 'detailled'):
            for i in np.arange(5):
                stim_in_run.append(genres[g] + '_00' + str(i))
        if (mode == 'by_genre'):
            for i in np.arange(5):
                stim_in_run.append(genres[g])

    ## crétion des labels de tous les sujets
    for s in np.arange(nsubs):
        for r in np.arange(8): # 8 runs par sujets
            labels += stim_in_run
    labels = pd.DataFrame.from_dict(labels)
    return(labels)

#############################################################################################################################
## fMRI Signal Extraction Functions ##################################################### fMRI Signal Extraction Functions ##
#############################################################################################################################

def roi_values_extraction(mask_img, subs, nsubs, betas, betas_path,  verbose = 1):
    """
    Extract signal values from .nii images using a mask defining Regions Of Interest

    Parameters :
      - mask_img : nii image of the mask from which the signal is extracted                     | type : nii 3D image
      - subs : list of the subject prensent in the 'betas' folder                               |  type list
      - nsubs : number of subject                                                               |  type int
      - betas : dictionnary containing key for each subject                                     |  type dict
                Each key leading to the list of the subject's betas
      - betas_path : path leading to the betas' folder (1 folder per subject folder within).    | type : str

    Output :
      - masked_imgs : array with the extracted values from each image                           |  type array of object
                      of each subject. It's an array of dimension M x N,
                      with M : the number of subjects and N : the number
                      of images for each subject. The cells are filled
                      with object : np.array
    """

    for s in np.arange(nsubs):
        sub_name = subs[s]
        ## Just display the current subject
        print('Sujet en cours : {0}' .format(sub_name))
        time.sleep(1)
        ## Extracting...
        masked_imgs = np.ndarray((12, 200) , dtype = object)
        for i in np.arange(len(betas[0])):
            beta_img = image.load_img(betas_path + '/' + sub_name + '/' + betas[s][i])
            masked_imgs[s][i] = masking.apply_mask(beta_img, mask_img)

        ## Refresh display
        if s < nsubs-1:
            print("\033[A \033[K \033[A")  #go at the begining of the previous line and clear it
        else:
            print("Extraction terminée")
    if verbose > 0:
        print('Nombre de voxels présents dans la ROI dont le signal a été extrait : {0}' .format(len(masked_imgs[0][0])))

    return(masked_imgs)

def save_roi_values(masked_imgs, save_path, subs, mask_name="masked"):
    """
    Save the values contained in an ROI extracted with the function : 'roi_values_extraction'.
    The data are saved under the HDF5 format with a pandas function. One file per subject is
    created at the indicated location
    If a file with the same name is present in the location, it will be deleted.

    Parameters :
        - masked_imgs : the values that have been extracted. The shape should be number of subjects x
                        number of image per subject.
        - save_path : the place where the files will be saved. Take care about this place MUST exist,
                      it will not be created.
        - subs : list of the subjects, used for the saving name.
        - mask_name : string indicating the name of the used mask. Used for the saving name of the files.
    """
    for s in np.arange(len(subs)):
        masked_imgs = pd.DataFrame.from_dict(masked_imgs)
        masked_imgs.to_hdf(save_path + '/' + subs[s] + '_' + mask_name + '.h5' , key = 'roi_values', mode = 'w')

    return

def NU_searchlight_values_extraction(mask_img, radius, subs, nsubs, betas, betas_path):
    """
    Extract signal values from .nii images using a main mask defining Regions Of
    Interest and several spherical masks centered in each voxel contained the main ROI

    Parameters :
      - mask_img : nii image of the mask from which the signal is extracted                     | type : nii 3D image
      - radius : in mm, the radius
      - subs : list of the subject prensent in the 'betas' folder                               |  type list
      - nsubs : number of subject                                                               |  type int
      - betas : dictionnary containing key for each subject                                     |  type dict
                Each key leading to the list of the subject's betas
      - betas_path : path leading to the betas' folder (1 folder per subject folder within).    | type : str

    Output :
      - masked_imgs : dictionnary containing matrix of array containing all the masked images values (np.array)
    """
    masked_values = {}
    for s in np.arange(nsubs):
        ### Chargement préliminaire des 200 images pour ne pas a avoir à les recharger à chaque voxel
        beta_img_nii_list = []
        for i in np.arange(len(betas[0])):
            beta_img_nii_list.append(image.load_img(betas_path + subs[s] + '/' + betas[s][i]))
        beta_img_nii = image.concat_imgs(beta_img_nii_list)

        searchlight = decoding.SearchLight(
            mask_img,
            radius=radius, n_jobs=-1,
            verbose=1)

        searchlight.compute_sphere(beta_img_nii)
        A = searchlight.sphere_vect
        X = searchlight.img_vect

        masked_values[s] = decoding.searchlight.get_spheric_mask_values(X,A)

    return masked_values

def NU_save_srcl_values(masked_values, save_path, subs):
    """
    Save the values contained in an ROI extracted with the function : 'roi_values_extraction'.
    The data are saved under the HDF5 format with a pandas function. One file per subject is
    created at the indicated location
    If a file with the same name is present in the location, it will be appended.

    Parameters :
        - masked_imgs : dictionnary containing matrix of array containing all the masked images values (np.array)
                        length of the dictionnary is the number of subjects and
                        shape of the values are : number of spheres x number of images
        - save_path : the place where the files will be saved. Take care about this place MUST exist,
                      it will not be created.
        - subs : list of the subjects, used for the saving name.
        - mask_name : string indicating the name of the used mask. Used for the saving name of the files.
    """

    for s in np.arange(len(subs)):

        to_save = pd.DataFrame(masked_values[s])
        print(to_save.shape)
        size = to_save.memory_usage(deep = 'True').sum()/(1000000000)
        print(size)
        ### Chunk the saving of the data if they exceed the 4GB maximum HDF5 norm.
        nbr_step = np.ceil(size/3.5)
        print(nbr_step)
        for i in np.arange(nbr_step):
            to_save.to_hdf(save_path + '/' + subs[s]+'_searchlight_1010values.h5', key='masked_values')
    return

def NU_save_srcl_values3(masked_values, save_path, subs):
    """
    Save the values contained in an ROI extracted with the function : 'roi_values_extraction'.
    The data are saved under the HDF5 format with a pandas function. One file per subject is
    created at the indicated location
    If a file with the same name is present in the location, it will be appended.

    Parameters :
        - masked_imgs : dictionnary containing matrix of array containing all the masked images values (np.array)
                        length of the dictionnary is the number of subjects and
                        shape of the values are : number of spheres x number of images
        - save_path : the place where the files will be saved. Take care about this place MUST exist,
                      it will not be created.
        - subs : list of the subjects, used for the saving name.
        - mask_name : string indicating the name of the used mask. Used for the saving name of the files.
    """

    for s in np.arange(len(subs)):

        to_save = pd.DataFrame(masked_values[s])
        test = to_save.take([0])
        print(type(test))
        print(test.shape)

        # for sph in np.arange(to_save.shape[0]):
        #     temp = to_save.take([0])
        #     temp.to_hdf(save_path + '/' + subs[s]+'_searchlight_values_10mm.h5', key=str(sph))
    return


def NU_save_srcl_values2(masked_values, save_path, subs):
    """
    Save the values contained in an ROI extracted with the function : 'roi_values_extraction'.
    The data are saved under the HDF5 format with a pandas function. One file per subject is
    created at the indicated location
    If a file with the same name is present in the location, it will be appended.

    Parameters :
        - masked_imgs : dictionnary containing matrix of array containing all the masked images values (np.array)
                        length of the dictionnary is the number of subjects and
                        shape of the values are : number of spheres x number of images
        - save_path : the place where the files will be saved. Take care about this place MUST exist,
                      it will not be created.
        - subs : list of the subjects, used for the saving name.
        - mask_name : string indicating the name of the used mask. Used for the saving name of the files.
    """

    for s in np.arange(len(subs)):

        to_save = pd.DataFrame(masked_values[s])
        print(to_save.shape)
        size = to_save.memory_usage(deep = 'True').sum()/(1000000000)
        print(size)
        ### Chunk the saving of the data if they exceed the 4GB maximum HDF5 norm.
        nbr_step = np.ceil(size/3.5)
        print(nbr_step)
        # for i in np.arange(nbr_step):

        base = pd.HDFStore('save_path' + '/' + subs[s]+'_searchlight_2121values.h5')
        base.append('test')
    return

def extract_spheres_coord(mask_img, radius, save_path = None, sufixe = None, saving = False):
    """
    This function is made for optimisation. It has the same principle as the same
    principle as the 'searchlight_values_direct_saving' function but it only compute
    the spheres' coordinates and indices for 1 subjects. Since some masks have a large
    amount of voxels and the radius of the spheres can be quite large, the computing
    of the spheres can be quite long and we don't want to launch this computation
    each time with the script.
    Instead, we prefer to save a sparse matrix (with coo format) of the spheres coordinates
    at .npz format that will be easily load and treated (coo matrix to lil matrix) to fit as
    a parameter of the function 'searchlight_values_direct_saving' with the funtion 'load_spheres_coord'

    Parameters =
        - mask_img : nii image of the mask from which the signal is extracted.
        - radius : define the radius of the spheres (in mm)
        - save_path : define the path where the matrix has to be saved
        - sufixe : add somethong to the saving name of the matrix. The name of the mask for instance
        - saving : activate or deactivate thee saving of the matrix.

    Outputs =
        - A : the sparse matrix containing the coordinates of the spheres and the indices of the voxels within it.
    """

    img = image.concat_imgs([mask_img])
    searchlight = decoding.SearchLight(
        mask_img,
        radius=radius, n_jobs=-1,
        verbose=1)
    searchlight.compute_sphere(img)
    A = searchlight.sphere_vect

    if saving :   # If the saving otion is activated
        to_save = A.tocoo()
        save_name = str(radius) + 'mm_spheres_coodinates'
        if sufixe != None :
            save_name += '_' + sufixe
        save_npz(save_path + '/' + save_name, to_save)

    return A

def load_spheres_coord(path_name) :
    """
    This function is usefull to load a sparse matrix saved by the 'extract_spheres_coord' function.
    The loaded matrix is in the .coo format, for using it in the 'searchlight_values_direct_saving' function,
    it needs to be convert as a LIL Matrix.

    Parameters :
        - path_name : path of the sparse matrix.
        -

    Outputs :
        - spheres_coord
    """

    A_coo = load_npz(path_name)
    A = A_coo.tolil()

    return A


import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

def searchlight_values_direct_saving(mask_img, radius, subs, nsubs, betas, betas_path, save_path, sufixe = None, chunk_size = 500, sph_coords = None):
    """
    This functions has the same goal than 'searchlight_values_extraction'
    but instead of keeping the masked
    value in memory, it will directly save them in hdf5 file.
    It has to be used when :
    - mermory is quite low on your computing machine
    - if the masked_values in the output is larger than it is possible
    to store them in hdf5 file in the direct way.

    All the values of a single subject will be stored in an unique hdf5 file
    but each sphere will be under a different key (corresponding the it number)

    Parameters :
      - mask_img : nii image of the mask from which the signal is extracted                     | type : nii 3D image
      - radius : in mm, the radius
      - subs : list of the subject prensent in the 'betas' folder                               |  type list
      - nsubs : number of subject                                                               |  type int
      - betas : dictionnary containing key for each subject                                     |  type dict
                Each key leading to the list of the subject's betas
      - betas_path : path leading to the betas' folder (1 folder per subject folder within).    | type : str
      - save_path : the place where the files will be saved. Take care about this place MUST exist,
                    it will not be created.
      - sufixe : the name of the output files will be define with the name of the considered sub and a sufixe.
                 the default sufixe is '_searchlight_[radius]_mm_values.h5'.
                 If a personnal sufixe is declared in the function call, the extension have to be had to the sufixe.
      - chunk_size : default 500. Size of each group in the saved hdf5 file.
      - sph_coords : Optionnaly, the corrdinate for the sphere can be compute before (with extract_spheres_coord function).
                     Since we use an unique mask, the spheres can be the same across the subjects and don't have to be computed each time.
    """
    if sufixe == None:
        sufixe = '_searchlight_' + str(radius) + '_mm_values.h5'
    for s in range(nsubs):
        ### Chargement préliminaire des 200 images pour ne pas a avoir à les recharger à chaque voxel
        beta_img_nii_list = []
        for i in range(len(betas[0])):
            beta_img_nii_list.append(image.load_img(betas_path + subs[s] + '/' + betas[s][i]))
        beta_img_nii = image.concat_imgs(beta_img_nii_list)

        searchlight = decoding.SearchLight(
            mask_img,
            radius=radius, n_jobs=-1,
            verbose=1)

        #### N'exécuter cette fonction que lors du premier sujet. Les sphères restant ls meme il n'est pas besoin de les recalculer.
        if s == 0 and sph_coords == None:       ### Si on traite le premier sujet et que les coordonnées des sphères ne sont pas données en entrée
            searchlight.compute_sphere(beta_img_nii)
            A = searchlight.sphere_vect
            X = searchlight.img_vect
        elif s == 0 and sph_coords != None:     ### Si on traite le premier sujet et que l'on a donner les sphère en entrée
            A = sph_coords
            X = masking.apply_mask(beta_img_nii, mask_img)
        elif s>0 :                              ### Si on traite un nouveau sujet avec les même sphères que les précédents
            #On ne change pas A mais seulement les images masquées
            X = masking.apply_mask(beta_img_nii, mask_img)
        # print(A.shape)
        # print(X.shape)
        # print(A[0])

        if chunk_size == 'max':
            stride = A.shape[0]     # We take all the centers at once
        else :
            stride = chunk_size


        save_name = '_' + str(radius) + 'mm_spheres'
        if sufixe != None :
            save_name += '_' + sufixe
        passe = 0
        print("Sujet en cours : ", subs[s])
        for sph in np.arange(0, A.shape[0] , stride):
            sys.stdout.write('Computing values ... Chunks : {0}/{1} \r' .format(passe+1 , int(np.ceil(A.shape[0]/stride))))
            masked_values = decoding.searchlight.get_spheric_mask_values(X,A[sph:sph+stride,:])
            # print(masked_values.shape)
            # store.put('values',masked_values, format='fixed', append = 'True' )
            to_save = pd.DataFrame(masked_values)
            with HiddenPrints():   # !!!!!!!!!!!!!!!! If an error occure in the saving, it won't be shown in stderr !!!!!!!!!!!!!!!!!!!!! #
                to_save.to_hdf(save_path + '/' + subs[s] + save_name + '.h5', key='chunk_'+str(passe))
            passe += 1
        print('')
    return

#############################################################################################################################
## Function Utilities for Treating the Data ##################################### Function Utilities for Treating the Data ##
#############################################################################################################################

# Whitening/demeaning...


#############################################################################################################################
## DSM Utilities ########################################################################################### DSM Utilities ##
#############################################################################################################################

def sort_stim(stims_values , mode = 'alpha'):
    """
    Fonction permettant de changer l'ordre des stimuli dont on va
    extraire les DSMs
    !!! stims_values doit avoir comme dimension NxM où N est le nombre de stimuli et M le nombre d'occurences !!!
    !!! 1 ligne = 1 stimulus !!!
    """
    ### Si le mode rentré est incorrect, on le réinitialise à 'alpha'
    if mode != 'alpha' and mode != 'guclu':
        print("Mode ", mode, " incorrect : valeur par défaut 'alpha' utilisée")
        mode='alpha'

    if mode=='alpha':
        pass    ### On ne fait rien car les stimuli sont déjà rangés par ordre alphabétique
    if mode=='guclu':
        ### Extraction de chaque genre différent
        ambiant = stims_values[0:5]
        country = stims_values[5:10]
        metal = stims_values[10:15]
        rocknroll = stims_values[15:20]
        symphonic = stims_values[20:25]
        #### on reforme la matrice total avec le meme ordre que dans le papier de Guclu
        sorted_stim = np.vstack((ambiant, symphonic, country, rocknroll, metal))

    return sorted_stim

def mean_over_runs(nbr_runs, values):
    """
    Fonction permettant de calsuler la moyenne des valeurs correspondant aux mêmes stimuli à travers 8 runs
    """
    l = int(values.shape[0] / nbr_runs) #Number of stimuli for 1 run

    means = np.ndarray((values[0].shape[0], l))

    for s in np.arange(l):   # On boucle sur les 25 stimuli
        stim_values = list()
        for r in np.arange(nbr_runs):   # On boucle sur les 8 runs
            stim_values.append(values[s + l*r])   # on prends la liste des 8 runs d'un meme stimulus
        # means[:, s] = np.mean( (values[s] , values[s+l] , values [s+(l*2)], values[s+(l*3)], values[s+(l*4)] , values[s+(l*5)] , values [s+(l*6)], values[s+(l*7)]) , axis = 0)
        means[:, s] = np.mean( stim_values, axis = 0)

    return means

def dsm_spearman(stims_values):
    """
    Take a np.array of dimension 2 with columns representing stimuli
    and row representing occurences
    """
    nbr_stim = stims_values.shape[0]

    # dsm = np.empty((nbr_stim,nbr_stim))

    ### Compute full Matrix
    # for i in np.arange(nbr_stim):
    #     for j in np.arange(nbr_stim):
    #         r,p = spearmanr(stims_values[i],stims_values[j])
    #         dsm[i, j] = 1 - r

    ### Compute triangular sup matrix
    # for i in np.arange(nbr_stim):
    #     for j in np.arange(nbr_stim - i -1 ):
    #         r,p = spearmanr(stims_values[i],stims_values[j+i+1])
    #         dsm[i, j+i+1] = 1 - r

    ### Compute triangular sup Matrix stored in  array
    length = int(((nbr_stim * nbr_stim) - nbr_stim)/ 2) #length of the upper side
    dsm_vect = np.empty((1 , length))
    ind = 0
    for i in np.arange(nbr_stim):
        for j in np.arange(nbr_stim - i -1 ):
            r,p = spearmanr(stims_values[i],stims_values[j+i+1])
            dsm_vect[0][ind] = 1 - r
            ind += 1

    return dsm_vect

def dsm_euclid(stims_values):
    """
    Take a np.array of dimension 2 with columns representing stimuli
    and row representing occurences
    """
    nbr_stim = stims_values.shape[0]

    ### Compute triangular sup Matrix stored in  array
    length = int(((nbr_stim * nbr_stim) - nbr_stim)/ 2) #length of the upper side
    dsm_vect = np.empty((1 , length))
    ind = 0
    for i in np.arange(nbr_stim):
        for j in np.arange(nbr_stim - i -1 ):
            d = euclidean(stims_values[i],stims_values[j+i+1])
            dsm_vect[0][ind] = d
            ind += 1

    return dsm_vect

## Manage DSMs ##
def save_dsms(dsms, save_path, save_name):
    to_save = pd.DataFrame(dsms)
    to_save.to_hdf(save_path + '/' + save_name , key = 'DSMs')

def load_dsms(path_name):

    dsms = pd.read_hdf(path_name).values

    return dsms

def vect2triu(dsm_vect):
    #On retrouve la dimension de la matice (onse sert de l'apporximation  :  sqrt(X²) =~= sqrt(X²-X)  soit  sqrt(X²) = ceil(sqrt(X²-X))   )
    dim = int(np.ceil(np.sqrt(dsm_vect.shape[1] * 2)))
    dsm = np.zeros((dim,dim))
    ind_up = np.triu_indices(dim,1)
    dsm[ind_up] = dsm_vect
    return dsm

def triu2full(dsm_triu):
    dsm_full = np.copy(dsm_triu)
    ind_low = np.tril_indices(dsm_full.shape[0], -1)
    dsm_full[ind_low] = dsm_full.T[ind_low]
    return dsm_full

def full2vect(dsm_full):
    dim = dsm_full.shape[0]
    ind_up = np.triu_indices(dim, 1)
    dsm_vect = np.zeros((1, int((dim*dim-dim)/2)))
    dsm_vect[0] = dsm_full[ind_up]
    return dsm_vect

def show_dsm(dsm):
    # fig = plt.figure()
    plt.imshow(dsm, cmap = 'jet');
    plt.colorbar()
    # plt.show()
    return

def save_png_dsm(dsm, saving_name, title = None):
    """This focntion take in input the DSM withe a square format
    (use vect2triu and triu2full)"""
    fig = plt.figure()
    plt.imshow(dsm, cmap = 'jet')
    plt.title(title)
    plt.colorbar()
    plt.savefig(saving_name)
    plt.close()

#############################################################################################################################
## First Level RSAnalysis (computing from a single model) ######### First Level RSAnalysis (computing from a single model) ##
#############################################################################################################################

def get_h5_info(h5_file):
    """
    This function det the path name of a hdf5 file and return the number
    of groups that are present in the file and the number of rows.

    This function must to be call when you save hdf5 files from
     searchlight_values_direct_saving and you don't remember/know
     how many spheres were saved and in how many groups the files has been chunked.

    """
    with pd.HDFStore(h5_file, mode='r') as store:
        # store = pd.HDFStore(h5_file)
        nbr_chunks = len(store.keys())
        # print('Nombre de chunks :', nbr_chunks)

        norm_chunk = store.get('chunk_0')
        # print('Shape des chunks complets :', norm_chunk.shape)
        nbr_spheres = norm_chunk.shape[0]  # If only 1 group, its lentgh is the total length

        if nbr_chunks > 1:
            ended_chunk = store.get('chunk_' + str(nbr_chunks-1))
            # print('Shape du dernier chunk :', ended_chunk.shape)
            nbr_spheres = norm_chunk.shape[0] * (nbr_chunks-1) + ended_chunk.shape[0]     # length of normal chunk * number-1 + lentgh of the last chunk

    return nbr_chunks, nbr_spheres, norm_chunk.shape[0]

def get_dsm_from_searchlight(h5_file, nbr_chunks, nbr_spheres, nbr_runs, metric = 'spearmanr'):
    """
    Usefull for brain model
    From a H5 file containing signal of several
    compute dissimilarity matrice of these condition.
    The dissimilarity can be computed with various metrics as euclidean distance or
    spearman (pearson ranked) correlation.
    """
    with pd.HDFStore(h5_file, mode='r') as store:
        count = 0
        dsm = np.empty((1, nbr_spheres), dtype='object')
        for ch in tqdm(np.arange(nbr_chunks)):  ## Loop on all the groups in the hdf5 file
            sph_values = store.select(key = 'chunk_' + str(ch)).values
            # print(sph_values.shape)
            for sph in np.arange(sph_values.shape[0]):
                means = mean_over_runs(nbr_runs = nbr_runs, values = sph_values[sph])
                means = sort_stim(np.transpose(means)  , mode = 'guclu')

                if metric == 'spearmanr':
                    dsm[0][count] = dsm_spearman(means)

                if metric == 'euclidean':
                    dsm[0][count] = dsm_euclid(means)
                count += 1
    return dsm

def get_dsm_from_searchlight_opti(h5_file, nbr_chunks, nbr_spheres, norm_chunk_length, nbr_runs, metric = 'spearmanr'):
    """
    Usefull for brain model
    From a H5 file containing signal of several
    compute dissimilarity matrice of these condition.
    The dissimilarity can be computed with various metrics as euclidean distance or
    spearman (pearson ranked) correlation.
    """
    with pd.HDFStore(h5_file, mode='r') as store:
        dsm = np.empty((1, nbr_spheres), dtype='object')
        for sph in np.arange(1000):#nbr_spheres):
            ch = sph//norm_chunk_length    #On selctionne le bon chunk
            # Seulement si on change de chunk, on charge les données du nouveau
            # et on met à jour le ch_prev
            if sph % norm_chunk_length == 0:#ch != ch_prev:
                sph_values = store.select(key = 'chunk_' + str(ch)).values
                ch_prev = np.copy(ch)
                # print(sph)

            means = mean_over_runs(nbr_runs = nbr_runs, values = sph_values[sph])
            means = sort_stim(np.transpose(means)  , mode = 'guclu')

            if metric == 'spearmanr':
                dsm[0][sph] = dsm_spearman(means)
            if metric == 'euclidean':
                dsm[0][sph] = dsm_euclid(means)
    return dsm

def get_dsm_from_searchlight_process(i, sph_values, norm_chunk_length, nbr_runs, chunk_size, metric = 'spearmanr'):
    """
    Usefull for brain model
    From a H5 file containing signal of several
    compute dissimilarity matrice of these condition.
    The dissimilarity can be computed with various metrics as euclidean distance or
    spearman (pearson ranked) correlation.
    """
    means = mean_over_runs(nbr_runs = nbr_runs, values = sph_values)
    means = sort_stim(np.transpose(means)  , mode = 'guclu')
    # dsms = np.empty((1, chunk_size), dtype='object')
    if metric == 'spearmanr':
        dsm = dsm_spearman(means)
    if metric == 'euclidean':
        dsm = dsm_euclid(means)
    return dsm

def get_dsm_parallel(h5_file, nbr_chunks, nbr_spheres, norm_chunk_length, nbr_runs, metric = 'spearmanr', n_jobs = 1):

    ### Open the hdf5 store
    with pd.HDFStore(h5_file, mode='r') as store:
        all_dsms = np.empty((1, 0), dtype='object') #create a no value array at wich the resulsts will be appended
        for ch in tqdm(range(nbr_chunks)):  ## Loop on all the groups in the hdf5 file
            sph_values = store.select(key = 'chunk_' + str(ch)).values
            chunk_size = sph_values.shape[0]
            # chunk_size = 33
            chunk_dsm = np.empty((1, chunk_size), dtype='object')
            results = Parallel(n_jobs=n_jobs, verbose = 0, backend = "multiprocessing")(
                delayed(get_dsm_from_searchlight_process)(
                    i, sph_values[i],  norm_chunk_length, nbr_runs,
                    chunk_size, metric = metric)
                for i in range(chunk_size))


            chunk_dsms = np.empty((1, chunk_size), dtype='object')
            chunk_dsms[0][:] = results
            # print(chunk_dsms.shape)
            # print(chunk_dsms[0][0].shape)
            all_dsms = np.append(all_dsms, chunk_dsms , axis = 1)
            # print(test[1][0].shape)
            # print(test.shape)
            #
            # print(np.asarray(results[0]).shape)

        #
        # print(type(all_dsms))
        # print(all_dsms.shape)
        #
        # print(all_dsms[0][0].shape)
        # print(all_dsms[0][50].shape)



    return all_dsms

#############################################################################################################################
## Second Level RSAnalysis (models comparisons) ############################# Second Level RSAnalysis (models comparisons) ##
#############################################################################################################################

def compare_dsms(DSM1,DSM2, verbose = 1, Norm = True):
    """
    Compare the DSM created from to different representationnal space
    (diffrent model of different values)
    Spearman correlation is generaly used for this comparison
    """

    ### Vérification de la forme vectorielle des DSMs
    if DSM1.shape[0] != 1 :
        if verbose >0:
            print('First matrix is not vectorized')
            print('Assuming the matrix is full, it is converted with nilearn.decoding.rsanlysis.full2vect()')
        DSM1 = full2vect(DSM1)

    if DSM2.shape[0] != 1 :
        if verbose >0:
            print('Second matrix is not vectorized')
            print('Assuming the matrix is full, it is converted with nilearn.decoding.rsanlysis.full2vect()')
        DSM2 = full2vect(DSM2)

    if Norm :  #Si il faut normaliser les DSMs
        ##DSM1
        # print(DSM1)
        DSM1 = DSM1/(max(DSM1[0]))
        # print(DSM1)
        DSM2 = DSM2/(max(DSM2[0]))
    ### Speramn r corrélation entre les matrices

    r, _ = spearmanr(DSM1, DSM2)
    return r

def NW_cand_dsms_to_one_ref(DSMs_candidate , DSM_reference):
  """
  The idea there is to take a "candidate" set of DSMs and a "reference" DSM.
  All candidates will be correlated to the reference and the most matching
  candidate will be saved.
  This saving take the form of and one-hot encoded array with the winning DSM
  candidate index.MD

  e.g. :
  7 candidates are compared to the reference.
  A first array is computed with the score of each candidate :
  [0.5, 0.1, 0.3, 0.9, 0.4, 0.4, 0.7]
  The second array one hot encode the first using the hghest score :
  [  0  , 0,   0,   1,   0,   0,   0]
  """

def cand_dsms_to_ref_dsms(DSMs_candidate , DSMs_reference , attribution_mode = 'auto', verbose = 1):
    """
    The idea there is to take 2 sets of DSMs. A "candidate" set and a "reference" one.
    All candidates will be correlated to all references and the maximum correspondances will be save.
    The finality of that is an attribution of the candidate dsms to the reference ones.

    This attribution take the form of an array 3xL with L dipendending on the 'attribution_mode' :
    - "on_reference" :  L is the reference set length.
                        The first row of the array is the index of the ref. DSMs.
                        Second row is the index of the cand. DSM attribuated to each ref.
    - "on_candidate" :  L is the candidate set length.
                        The first row of the array is the index of the cand. DSMs.
                        Second row is the index of the ref. DSM on which each cand has been attribuated.
    - "auto"         :  compare the length of the both sets and :
                        - act like "on_reference" mode if ref. set is the largest
                        - act like "on_candidate" mode if cand. set is the largest

    Third row is the correlation score between the candidate and the reference

    Take care about the attribution_mode : if L is taken from the smallest set, the second row will no more
    be composed by scalar but by arrays that will modify the output dtype!
    If you don't know or you are not sure about the length of the input sets, consider using 'auto' mode
    of the input sets, consider using 'auto' mode

    """

    #### Vérification du mode utilisé

    if attribution_mode == 'on_reference':
        print('This mode is not implemented yet ---> Default mode set')
        attribution_mode = 'auto'
    if attribution_mode == 'on_candidate':
        print('This mode is not implemented yet ---> Default mode set')
        attribution_mode = 'auto'
    if attribution_mode != 'on_reference' and attribution_mode != 'on_candidate' and attribution_mode != 'auto':
        print('Invalid mode ---> Default mode set')
        attribution_mode = 'auto'

    if attribution_mode == 'auto':
        if verbose >0 :
            print('Chosen mode : {}' .format(attribution_mode))
        L = max(DSMs_candidate.shape[1], DSMs_reference.shape[1])
        l = min(DSMs_candidate.shape[1], DSMs_reference.shape[1])

    #### Assigement array
    assig_array = np.ndarray((3, L))
    assig_array[0] = np.arange(L)
    # print(assig_array)
    for ref in tqdm(range(L)):   #range(L) 1 for test
        temp_scores = []
        for can in range(l):
            temp_scores.append(compare_dsms(DSMs_candidate[0][can], DSMs_reference[0][ref]))
        maxi = max(temp_scores)   #on cherche le max entre les scores
        ind = np.where(temp_scores == maxi)
        if len(ind) > 1: #On traite le cas de plusieurs maxima
            # ind = ind[0][-1]  #On prends la dernière des dsms candidates
            ind = random.choice(ind[0]) #On prends une dsms candidate au hasard
        elif len(ind) == 1 :
            try :
                ind = ind[0][0]
            except :
                print("Erreur sur le voxel :", ref)
                ind = 0
        assig_array[1][ref] = ind
        assig_array[2][ref] = maxi

    return assig_array

def correlation_map(DSM_candidate , DSMs_reference, Norm = True):
    """
    This function do the correlation between 1 DSM and a set of DSMs. Each score is return in an array

    Parameters :
        - DSM_candidate : Only 1 DSM that will be compared
        - DSMs_reference : Set of DSMs with which the candidate will be compared
        - Norm : define if a normalisation of the DSM is needed. If True, the normalisation between 0 and 1 will be done on ALL DSMs
    Output :
        - correlation_map_array :   Array with shape (2,L) with L the length of DSMs_reference
                                    Firts row is the indices of the ref and second row is the correlation score
    """

    L = DSMs_reference.shape[1]
    correlation_maps_array = np.empty((2, L))
    correlation_maps_array[0] = np.arange(L)
    for i in tqdm(range(L)):
        correlation_maps_array[1][i] = compare_dsms(DSM_candidate, DSMs_reference[0][i])

    return correlation_maps_array
