import copy
import random
import pandas as pd
import itertools
import shutil
import numpy as np 
import os
from os import listdir, makedirs
from os.path import isfile, isdir, join, exists
from tools.globals import GlobalVariables
from scipy.interpolate import interp1d
from scipy.optimize import minimize, nnls
from sklearn.metrics import explained_variance_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Union, Optional

dir_plots = 'plots'  # folder for exported visualizations
dir_trained_models = 'trained_models'  # folder for trained models
dir_data = 'data'  # folder for training/test data
dir_numpy_raw = join(dir_data, '1_numpy_raw')  # folder for .np data (before preprocessing)
dir_numpy_ready = join(dir_data, '2_numpy_ready')  # folder for .np data (after preprocessing)



def save_list_of_dicts(dir_data_out:str | os.PathLike, data:List[Dict], name:str):
    """
    Saves spectral data (list of dicts) to the specified directory (dir_data_out).

    :param dir_data_out: directory where data is saved (relative path)
    :type dir_data_out: str | os.PathLike
    :param data: spectral data (list of dicts, each dict contains one spectrum)
    :type data: List[Dict]
    :param name: filename of the saved data
    :type name: str
    """
    np.save(join(dir_data_out, name), data)


def load_spectral_data(dir_input:str|os.PathLike, detectors:List[str], **kwargs) -> Tuple[List[Dict], List[str]]:
    """
    Loads all spectra of input directory but only of the specified detector. 
    Special case: for 'simulated+bg', all simulated data and backgrounds (from measured data) are loaded.
    If exclude_files is given as a list, those files are not loaded.
    If include_files is given as a list, only those files are loaded from the directory.

    :param dir_input: directory from where data is loaded
    :type dir_input: str | os.PathLike
    :param detectors: specify from which detectors the data should be. 
            e.g. ['left', 'right', 'simulated', 'simulated+bg'] (or subset)
    :type detectors: List[str]

    :return: data_new: all spectral data (list of dictionaries)
            isotopes_det: 
    :rtype: Tuple[List[Dict], List[str]]
    """

    exclude_files = kwargs.get('exclude_files', [])  # optional parameter: files to be excluded
    include_files = kwargs.get('include_files', [])  # optional parameter: files to be included
    if include_files:
        all_names = sorted([x for x in listdir(dir_input) if x in include_files])
    else:
        all_names = sorted([x for x in listdir(dir_input) if x not in exclude_files])
    data_new = []
    if 'simulated+bg' in detectors:
        data_bg = np.load(join(dir_input, 'background.npy'), allow_pickle=True)
        data_new.extend(data_bg)
        detectors = ['simulated' if det=='simulated+bg' else det for det in detectors]
    for name in all_names:  # iterates over files in dir_input
        for det in detectors:
            data = np.load(join(dir_input, name), allow_pickle=True)
            data_from_det = [x for x in data if x['detector']==det]  # only spectra of specified detector
            data_new.extend(data_from_det)
    isotopes_det = sorted([sorted(y) for y in set([tuple(x['labels']) for x in data_new])])
    if len(data_new) == 0: 
        raise TypeError(f'Could not find data from {dir_input}, {detectors=} with {kwargs=}!')
    
    # save number of channels to global variable
    GlobalVariables.n_channels = len(data_new[0]['spectrum'])

    return data_new, isotopes_det


def limit_length_of_dataset(data:List[Dict], n_spectra_max:int) -> List[Dict]:
    """
    Option to limit the number of spectra dictionaries in a dataset (e.g. for quick tests). 
    If data is shorter than n_spectra_max, the complete list will be returned

    :param data_list: _list that should be shortened
    :type data_list: List[Dict]
    :param n_spectra_max: maximum number of elements in the output list
    :type n_spectra_max: int

    :return: shortened list (maximum length: n_spectra_max)
    :rtype: List[Dict]
    """
    
    if len(data) > n_spectra_max:
        return random.sample(data, n_spectra_max)  # randomly sample n_max elements
    else:
        return data  # return the full list if it's shorter than or equal to n_max
    


def cosine_similarity(a:Union[List[float], np.ndarray], b:Union[List[float], np.ndarray]) -> float:
    """
    Calculates the cosine similarity between two vectors (two spectra) with mean centering

    :param a: Spectrum a (list or numpy array)
    :type a: Union[List[float], np.ndarray]
    :param b: Spectrum b (list or numpy array)
    :type b: Union[List[float], np.ndarray]

    :return: cosine similarity between a and b
    :rtype: float
    """

    a = a - np.mean(a)  # mean centering of spectrum a
    b = b - np.mean(b)  # mean centering of spectrum b
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def calc_mean(isotope:str, folder:str|os.PathLike, detectors:List[str], norm:bool):
    """
    Opens the spectral data of isotope from folder and separates it into training and test set.
    Calculates the mean spectrum and standard deviation of the specified detectors for the training set.
    For datasets containing both isotope+background and pure background spectra: separates them 
    and calculates the means separately. 
    If norm=True, all means are normalized to integral 1 and standard deviations are adapted accordingly.

    :param isotope: isotope of which data are used, without file extension, e.g. 'Am241'
    :type isotope: str
    :param folder: directory where file is stored
    :type folder: str | os.PathLike
    :param detectors: list of detectors from which data are used, 
        e.g. ['right', 'left', 'simulated', 'simulated+bg'] (or subset of those)
    :type detectors: List[str]
    :param norm: Option to normalize the returned mean spectra to integral 1
    :type norm: bool

    :return: mean spectra and standard deviations (length: n_features) of the data found in <folder>/<name>, 
        separated into isotope and pure background. 
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """

    # if 'simulated+bg' is chosen as detector, it is changed to 'simulated' for simplicity
    detectors = [x.replace('simulated+bg', 'simulated') for x in detectors]  

    data = np.load(join(folder, f'{isotope}.npy'), allow_pickle=True)  # imports a list of dicts
    n_channels = len(data[0]['spectrum'])
    if isotope != 'background':
        data = [s for s in data if s['detector'] in detectors]  # keep only data of specified detectors

    # pure background spectra
    bg_spectra = np.array([s['spectrum'] for s in data if 'background' in s['labels'] and len(s['labels'])==1])

    # spectra containing the isotope
    iso_spectra = np.array([s['spectrum'] for s in data if np.any([lab in isotope for lab in s['labels']])]) 

    if len(iso_spectra) > 0:
        iso_train, iso_test = split_train_test(iso_spectra)
        mean_iso = np.mean(iso_train, axis=0, dtype=float)
        std_iso = np.std(iso_train, axis=0, dtype=float)
    else: 
        print(f'No isotope spectra were found for {isotope}, {detectors}. Returning array of zeros.')
        mean_iso = np.zeros(n_channels)
        std_iso = np.zeros(n_channels)

    if len(bg_spectra) > 0: 
        bg_train, bg_test = split_train_test(bg_spectra)
        mean_bg = np.mean(bg_train, axis=0, dtype=float)
        std_bg = np.std(bg_train, axis=0, dtype=float)

    else:  # e.g. for simulated spectra: retrieve background spectra from pure background file
        print(f'No pure background spectra found for {isotope}, {detectors=}, using background.npy instead.')
        pure_bg_data = np.load(join(dir_numpy_ready, 'background.npy'), allow_pickle=True)
        pure_bgs = [d['spectrum'] for d in pure_bg_data]
        mean_bg = np.mean(pure_bgs, axis=0, dtype=float)
        std_bg = np.zeros(n_channels)

    if norm:  # option to normalize the output spectra
        # normalize mean and standard deviation for isotope
        int_mean_iso = np.sum(np.abs(mean_iso))
        if int_mean_iso > 0:
            mean_iso = mean_iso / int_mean_iso
            std_iso = std_iso / int_mean_iso
        else: 
            print(f'No isotope spectra found for {isotope}, cannot normalize them!')
        
        # normalize mean and standard deviation for background
        int_mean_bg = np.sum(np.abs(mean_bg))
        if int_mean_bg > 0:
            mean_bg = mean_bg / int_mean_bg
            std_bg = std_bg / int_mean_bg
        else:
            print(f'No background spectra found for {isotope}, cannot normalize them!')
    return mean_bg, std_bg, mean_iso, std_iso 


def count_spectra_per_isotope(data:List[Dict]) -> List[int]:
    """
    Counts the number of spectra per isotope (aggregated by labels).
    Returns list of counts

    :param data: spectral data (list of dicts, each dict containing a spectrum, labels and metadata)
    :type data: List[Dict]
    """
    isotopes = sorted(set([tuple(sorted(x['labels'])) for x in data]))
    isotopes = [list(x) for x in isotopes]
    list_counts = []
    for iso in isotopes: 
        data_iso = [x for x in data if x['labels']==iso]
        print(f'{iso}: {len(data_iso)} spectra')
        list_counts.append(len(data_iso))
    return list_counts


def remove_empty_or_negative_spectra(data:List[Dict]) -> List[Dict]:
    """
    Checks spectral data for empty spectra or negative values. 
    Empty spectra are removed, negative values are replaced with 0. 

    :param data: spectral data (list of dicts, each dict containing a spectrum, labels and metadata)
    :type data: List[Dict]
    :raises ValueError: data has to include at least one element. 

    :return: spectral data, with empty spectra removed and negative values replaced by 0. 
    :rtype: List[Dict]
    """
    if len(data) == 0:
        raise ValueError(f'You passed an empty list as data! ')
    spectra = np.array([x['spectrum'] for x in data])

    spectra[spectra < 0] = 1.e-6  # replace negative values with (approx.) zero
    for da, spec in zip(data, spectra):
        da['spectrum'] = spec  # overwrite original spectra with non-negative spectra

    int_spectra = np.sum(np.abs(spectra), axis=1)
    data_new = [x for x, inte in zip(data, int_spectra) if inte > 0.]  # remove empty spectra
    return data_new


def check_for_nan_and_inf(variable_dict:Dict[str, np.ndarray], function_name:str):
    """
    Checks the value of variable_dict for NaNs and infs. 

    :param variable_dict: dictionary in the form {<variable name>: <variable data>}, 
                            e.g. {'loadings': loadings_train}
    :type variable_dict: Dict[str, np.ndarray]
    :param function_name: name of the function where this function was called (for debugging)
    :type function_name: str
    """
    for var_name, var_value in variable_dict.items():

        if np.any(np.isnan(var_value)):
            print(f'{function_name}: NaN found value in {var_name}!')
        if np.any(np.isinf(var_value)):
            print(f'{function_name}: inf found in {var_name}!')


def split_train_test(array:Union[List, np.ndarray]) -> Tuple[Union[List, np.ndarray], Union[List, np.ndarray]]:
    """
    Splits data into a training and a test set. As test size and random state are fixed, this function ensures reproducibility 

    :param array: list or array to be split into training and test set
    :type array: Union[List, np.ndarray]

    :return: training and test set
    :rtype: Tuple[Union[List, np.ndarray], Union[List, np.ndarray]]
    """
    testsize = 0.2
    rand = 1
    array_tr, array_te = train_test_split(array, test_size=testsize, random_state=rand)
    return array_tr, array_te


def normalize_spectra(spectra:Union[List, np.ndarray]) -> Union[List, np.ndarray]:
    """
    Normalizes one spectrum or multiple spectra to integral 1. If spectra contains an empty spectrum, an error 
    message is thrown and the original spectra are returned.

    :param spectra: one or more spectra
    :type spectra: Union[List, np.ndarray]

    :return: normalized spectra (integral 1)
    :rtype: Union[List, np.ndarray]
    """
    
    if type(spectra) == List:  # check if labels is a nested list
        multiple_spectra = True if any(isinstance(item, list) for item in spectra) else False  
    
    elif type(spectra) == np.ndarray:
        multiple_spectra = True if spectra.ndim > 1 else False

    if multiple_spectra: 
        int_spectra = np.sum(np.abs(spectra), axis=1)  # calculate integrals 
        if np.all(int_spectra > 0):
            spectra_norm = np.divide(spectra, np.tile(int_spectra, (spectra.shape[1], 1)).T)
            return spectra_norm
        else: 
            i = np.where(int_spectra==0)
            print(f'ERROR: Cannot normalize spectra, integral of spectrum {i} is 0! Returning original spectra...')
            return spectra
    
    else:  # only one spectrum
        spectra_norm = spectra / np.sum(np.abs(spectra))
        return spectra_norm
    