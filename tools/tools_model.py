from tools.util import *
from tools.globals import GlobalVariables

def train_dim_model(data_tr:List[Dict], save_results:bool):
    """
    Trains a model for dimensionality reduction. 
    The model is a transformation matrix (loadings), a np.ndarray of shape n_isotopes x n_channels. 
    It consists of stacked mean spectra of isotopes_tr, only from detectors specified in dets_tr. 
    
    You should use only single-label spectra for training (e.g. pure isotope spectra without background). 
    Hence, if your measured spectra from dets_measured contain background, you should choose dets_tr = ['simulated+bg'].

    The loadings can be saved to dir_trained_models and loaded for application in spectrum analysis. 

    :param data_tr: training dataset (list of dictionaries with each dict containing one spectrum)
    :type data_tr: List[Dict]
    :param save_results: option to save the results to dir_trained_models
    :type save_results: bool
    """

    loadings_tr = [] 
    print(f'Building loadings from mean spectra of those isotopes: {GlobalVariables.isotopes_tr}')

    dets_train = copy.deepcopy(GlobalVariables.dets_tr)
    for iso in GlobalVariables.isotopes_tr:
        
        if 'simulated+bg' in dets_train:  # use simulated spectra and (measured) pure background spectra
            
            if iso == 'background':  # measured background spectra
                iso_spectra = [x['spectrum'] for x in data_tr if iso in x['labels']] 

            else:  # simulated spectra of isotope iso 
                iso_spectra = [x['spectrum'] for x in data_tr if [iso] == x['labels'] and any(x['detector'] in d for d in dets_train)]
        
        else:  # spectra from all other detectors, e.g. 'right', 'left' or 'simulated' of isotope iso
            iso_spectra = [x['spectrum'] for x in data_tr if [iso] == x['labels'] and x['detector'] in dets_train]
        
        mean_iso = np.mean(np.array(iso_spectra), axis=0, dtype=float)
        int_mean_iso = np.sum(np.abs(mean_iso[GlobalVariables.min_channel_tr:]))
        
        # normalize each mean spectrum to integral 1
        if int_mean_iso == 0.:
            raise ValueError(f'ERROR: Cannot normalize mean spectrum of {iso}! Integral: {int_mean_iso}')
        else:
            mean_iso = mean_iso / int_mean_iso
            loadings_tr.append(mean_iso)  # save to loadings_tr            

    loadings_tr = np.array(loadings_tr)  # convert to array

    check_for_nan_and_inf({'loadings': loadings_tr}, 'train_dim_model')

    GlobalVariables.loadings_tr = loadings_tr


    if save_results:  # option to save as np.ndarray in dir_trained_models
        savename = join(dir_trained_models, f'trained_dim_model')
        print(f'The trained model is saved to {savename}.')
        np.save(savename, loadings_tr)
 


def transform_spectra(data:List[Dict]) -> List[Dict]:
    """
    Calculates non-negative scores for all spectra by fitting it with non-negative-least-squares fit 
    to the loadings matrix. This corresponds to a dimensionality reduction: Spectra of shape n_measurements x n_channels 
    are converted to scores of shape n_measurements x n_isotopes_tr. Hence, in latent space, each dimension 
    corresponds to one isotope. 

    :param data: spectral data (list of dictionaries with each dict containing one spectrum)
    :type data: List[Dict]

    :return: spectral data with additional properties 'scores' (not normalized) and 'scores_norm' (normalized)
    :rtype: List[Dict]
    """

    # extract spectra from dictionaries
    spectra = np.array([x['spectrum'] for x in data])

    # load loadings (trained model) and min_channel from global variables
    loadings_tr = GlobalVariables.loadings_tr
    min_channel = GlobalVariables.min_channel_tr

    check_for_nan_and_inf({'loadings': loadings_tr, 'spectra': spectra}, 'transform_spectra()')

    # calculate scores (channels lower than min_channel_tr are ignored)
    for da, spec in zip(data, spectra):  # iterate over spectra

        # perform non-negative least squares fit between spectrum and loadings_tr
        coeffs, _ = nnls(loadings_tr.T[min_channel:, :], spec[min_channel:])
        da['scores'] = coeffs  # save as new property in dictionary

        coeffs_norm = coeffs / np.sum(np.abs(coeffs))  # scale to sum 1
        da['scores_norm'] = coeffs_norm  # save as new property in dictionary

    return data



def calculate_explained_variance(original:Union[List[float], np.ndarray], denoised:Union[List[float], np.ndarray]) -> float:
    """
    Calculates the explained variance ratio between original and denoised spectrum or spectra. 
    Channels below min_channel_tr are neglected.

    :param original: original spectrum or spectra
    :type original: Union[List[float], np.ndarray]
    :param denoised: denoised spectrum or spectra
    :type denoised: Union[List[float], np.ndarray]

    :return: explained variance ratio between original and denoised spectrum or spectra, between 0 and 1.
    :rtype: float
    """

    if original.ndim == 1:  # only one spectrum
        exp_var = explained_variance_score(original[GlobalVariables.min_channel_tr:], denoised[GlobalVariables.min_channel_tr:])
    
    if original.ndim == 2:  # multiple spectra
        exp_var = explained_variance_score(original[:, GlobalVariables.min_channel_tr:].T, denoised[:, GlobalVariables.min_channel_tr:].T)
    
    return exp_var


def denoise_spectra(data:List[Dict]) -> Tuple[List[Dict], float]:
    """
    Denoises spectra by transforming the scores back from latent space to spectral space. 
    In this step, noise will be lost. 
    
    Any spectrum with n_channels can be an argument but only spectra of isotopes in isotopes_tr will be denoised correctly.

    :param data: spectral data (list of dictionaries with each dict containing one spectrum)
    :type data: List[Dict]

    :return: data: spectral data with additional property 'denoised'
             explained_variance: explained variance ratio (value between 0 and 1) 
    :rtype: Tuple[List[Dict], float]
    """
    
    spectra = np.array([x['spectrum'] for x in data])  # extract spectra from data
    scores = np.array([x['scores'] for x in data])  # extract scores from data

    denoised_spectra = np.dot(scores, GlobalVariables.loadings_tr) # transform scores back to denoised spectra

    explained_variance = calculate_explained_variance(spectra, denoised_spectra)  # calculate explained variance
    
    for da, denoised_spectrum in zip(data, denoised_spectra):
        da['denoised'] = denoised_spectrum  # save as new property in data dictionary

    return data, explained_variance



def apply_dim_model(data:List[Dict]) -> List[Dict]: 
    """
    Applies the dimensionality reduction model to spectral data by transforming spectra into latent
    space (scores) and back (denoised spectra). 

    :param data: spectral data (list of dictionaries with each dict containing one spectrum and metadata)
    :type data: List[Dict]

    :return: spectral data with additional properties 'scores', 'scores_norm' and 'denoised'
    :rtype: List[Dict]
    """

    # transform spectra into latent space (calculate scores)
    data = transform_spectra(data)

    # transform scores back into spectral space (calculate denoised spectra)
    data, expl_var = denoise_spectra(data) 

    # print the explained variance ratio between original and denoised spectra (mean over whole dataset)
    print(f'Explained variance ratio: {expl_var: .1%} \n')

    return data






def create_prediction_dict(data_te:List[Dict]) -> List[Dict]:
    """
    Creates dictionary in the form {isotope: score} between isotopes_tr and scores, sorted by descending scores. 
    Background is moved to the end of the dictionary. 

    :param data_te: test dataset (list of dictionaries with each dict containing one spectrum and metadata)
    :type data_te: List[Dict]
    
    :return: data_te: test dataset with the additional property 'labels_pred_dict'
    :rtype: List[Dict]
    """
    for x in data_te:  # iterate over spectra
        scores = np.array(x['scores_norm'])
   
        # create dictionary for the predicted isotopes and their scores (for threshold min_scores_norm)
        labels_pred_dict = {iso: round(sco, 3) for iso, sco in zip(GlobalVariables.isotopes_tr, scores) \
                            if sco > GlobalVariables.min_scores_norm}

        # sort dictionary by scores (descending)
        labels_pred_dict_sort = sorted(labels_pred_dict.items(), key=lambda item: item[1], reverse=True)

        # change dictionary order (move 'background' to the end)
        labels_pred_dict_sort = [item for item in labels_pred_dict_sort if item[0] != 'background']  # all except background
        background_entry = [(k, v) for k, v in labels_pred_dict.items() if k == 'background']  # only background
        labels_pred_dict_sort.extend(background_entry)  # merge together (background at the end of list)
        labels_pred_dict_sort = dict(labels_pred_dict_sort)  # convert back to dictionary
        
        x['labels_pred_dict'] = labels_pred_dict_sort  # save dictionary to data_te
    
    return data_te


def print_multi_label_accuracies(data_te:List[Dict]):
    """
    Calculates the accuracies of multi-label predictions (compared to true labels). 
    1. Mean accuracy: One value reflecting the overall accuracy. Correct predictions contribute with their 
    positive score, incorrect predictions with their negative score. In addition, missing (true) labels are 
    penalized by subtracting their share from the scores. 

    2. Individual accuracies: 
    a) perfect predictions: predicted labels = true labels
    b) incomplete, but only-correct predictions: predicted labels < true labels
    c) complete, but partially wrong predictions: predicted labels > true labels
    d) incomplete and partially wrong predictions: predicted labels include wrong labels and some true labels are missing. 


    :param data_te: _description_
    :type data_te: List[Dict]
    """

    # sort labels alphabetically for each spectrum (test data)
    labels_te = [sorted(x['labels']) for x in data_te]

    # 1. Calculate the mean multi-label accuracy 
    labels_pred = []  # list for predicted labels    
    accuracies = []  # list for accuracies of the predictions

    for x in data_te:

        # extract predicted and true labels as lists
        labels_pred_i = list(x['labels_pred_dict'].keys())
        labels_true_i = x['labels'] 
        
        # calculate accuracies: positive score for correct and negative scores for incorrect predictions
        acc_list = [sco if iso in x['labels'] else -sco for iso, sco in x['labels_pred_dict'].items()]

        # add penalties for missing true labels
        acc_list.extend([-1./len(labels_true_i) for iso_true in labels_true_i if iso_true not in labels_pred_i])

        # sum all elements of acc_list, minimum is 0
        acc = np.max([0., np.sum(acc_list)])

        labels_pred.append(labels_pred_i)  # save predicted isotopes to list
        accuracies.append(acc)  # save accuracy to list
    
    # calculate mean accuracy
    mean_accuracy = np.mean(accuracies)
    print(f'Mean multi-label-accuracy: {mean_accuracy: .1%} \n')


    # 2. Calculate individual accuracy types (perfect, incomplete, partially wrong, incomplete & partially wrong)
    # a) perfect predictions
    n_perf = len([x for x, y in zip(labels_te, labels_pred) if x==y])
    ratio_perf = n_perf/len(data_te)
    print(f'{n_perf} perfect multi-label-predictions ({ratio_perf: .1%}). ')

    # b) incomplete, but only-correct predictions: predicted labels < true labels
    n_incomplete = len([x for x, y in zip(labels_te, labels_pred) if set(y)< set(x) and x!=y])
    ratio_incomplete = n_incomplete/len(data_te)
    print(f'{n_incomplete} incomplete, but only-correct multi-label-classifications ({ratio_incomplete: .1%}). ')

    # c) complete, but partially wrong predictions: predicted labels > true labels
    n_wrong = len([x for x, y in zip(labels_te, labels_pred) if set(x) < set(y) and x!=y])
    ratio_wrong = n_wrong/len(data_te)
    print(f'{n_wrong} complete, but partially wrong multi-label-classifications ({ratio_wrong: .1%}). ')

    # d) incomplete and partially wrong predictions: predicted labels include wrong labels and some true labels are missing. 
    n_wrong_and_incomplete = len([x for x, y in zip(labels_te, labels_pred) if not set(x) < set(y) and not set(y) < set(x) and x!=y])
    ratio_wrong_incomplete = n_wrong_and_incomplete/len(data_te)
    print(f'{n_wrong_and_incomplete} incomplete, partially wrong multi-label-classifications ({ratio_wrong_incomplete: .1%}). ')

    
def print_single_label_accuracy(data_te: List[Dict]):
    """
    Calculates and prints the accuracy of the single-label predictions. 
    This means that for predicted labels, only the isotope with the highest score counts.
    For true labels (where we have no information on the contribution), the first isotope counts, 
    but isotopes are preferred to background. 

    :param data_te: test dataset (list of dictionaries with each dict containing one spectrum and metadata)
    :type data_te: List[Dict]
    """

    # extract true labels and keep only first element (prefer isotopes to background)
    labels_te = [x['labels'] for x in data_te]
    labels_te_flat = [next((x for x in lab if x != 'background'), lab[0]) for lab in labels_te]

    # extract predicted labels and keep only first element (with highest score)
    labels_pred = [list(x['labels_pred_dict'].keys())[0] for x in data_te]

    # calculate accuracy of single-label prediction
    acc = accuracy_score(labels_te_flat, labels_pred)
    print(f'Single-label classification accuracy: {acc: .1%}')


def classify_from_scores(data_te:List[Dict], class_type:str) -> List[Dict]:
    """
    Uses the scores to predict which isotopes are found in each test spectrum. 
    First, a dictionary is created in the form {isotope: score}, ordered by descending scores. 
    Second, the accuracy of the prediction over the whole test dataset is calculated. 
    Options: 
    - single-label classification: compare only first true/predicted label (for single-label spectra)
    - multi-label classification: compare all true/predicted labels (for multi-label spectra)

    :param data_te: test dataset (list of dictionaries with each dict containing one spectrum and metadata)
    :type data_te: List[Dict]
    :param class_type: can be 'single-label' or 'multi-label'
    :type class_type: str

    :return: test dataset with additional property ['labels_pred_dict'] (dictionary of predicted 
            isotope-score pairs)
    :rtype: List[Dict]
    """

    # order scores by size and add dictionary with predicted isotope names and scores to each spectrum
    data_te = create_prediction_dict(data_te)

    if class_type == 'single-label':  # keep only the first entry of the dictionary (highest score)
        print_single_label_accuracy(data_te)
        
    elif class_type == 'multi-label':  # compare all predicted labels with all true labels
        print_multi_label_accuracies(data_te)

    else:
        raise ValueError(f'class_type can be single-label or multi-label but you chose {class_type}!')
    return data_te


 