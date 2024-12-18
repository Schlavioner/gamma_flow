from scipy.optimize import curve_fit
import sklearn as skl
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score

from tools.tools_model import *
from tools.globals import GlobalVariables


def enrich_dicts_with_information(list_dict:List[Dict]):
    """based on available information of each spectra, additional information is calculated and stored.

    :param list_dict: list of dictonaries containging spectra and meta data
    :type list_dict: List[Dict]
    :return: same dicts with additional information
    :rtype: List[Dict]
    """
    for da in list_dict:
        da["scores_norm_sorted"] = sorted(da["scores_norm"], reverse=True)
        da["scores_norm_mean"] = np.mean(da["scores_norm"])
        da["scores_norm_median"] = np.median(da["scores_norm"])
        da["explained_variance"] = explained_variance_score(da["spectrum"], da["denoised"])
        da["cosine_similarity"] = cosine_similarity(da["spectrum"], da["denoised"])
        da["scores_norm_abs"] = np.sqrt(np.sum(np.square(da["scores_norm"])))
    return list_dict


def split_spectra_one_vs_all(data:List[Dict], isotopes:List[str], excluded_isotope:str):
    """
    Split data by label into two datasets: one isotope and all others.

    :param data: spectral data
    :type data: List[Dict]
    :param isotopes: list of all isotopes found in data as labels
    :type isotopes: List[str]
    :param excluded_isotope: _description_
    :type excluded_isotope: str

    :return: data_one: data of excluded_isotope
            isotope_one: name of excluded_isotope (as array)
            data_other: data from all isotopes but excluded_isotope
            isotopes_other: names of all isotopes but excluded_isotopes (as array)
    :rtype: Tuple
    """
    
    if excluded_isotope not in isotopes:
        raise ValueError(f'You are trying to remove {excluded_isotope} from the data but \
                         it does not exist! Given isotopes: {isotopes}')
   
    # divide isotopes into two arrays: 1. excluded_isotope, 2. all others
    isotope_one = np.array([excluded_isotope])
    isotopes_other = np.array([iso for iso in isotopes if iso != excluded_isotope])
    print(f'\n Excluded isotope: {isotope_one}')

    data_one = [x for x in data if excluded_isotope in x['labels']]  # all data of excluded_isotope
    data_other = [x for x in data if excluded_isotope not in x['labels']]  # all data of other isotopes

    return data_one, isotope_one, data_other, isotopes_other


def simulate_outlier(isotope_outlier:str):
    """
    Trains a dimensionality reduction model with all outliers except for isotope_outlier.
    Then, the model is applied to all known data, but also to unknown data (of isotope_outlier).
    We record the information how unknown spectra appears in latent space
    and construct two arrays with information on the known and unknown isotopes.

    :param isotope_outlier: name of isotope that should be excluded from training
    :type isotope_outlier: str
    :return: np.array(), np.array()
    """

    data, all_isotopes = load_spectral_data(dir_numpy_ready, GlobalVariables.dets_tr)
    all_isotopes = [x[0] for x in all_isotopes]  # only single-label spectra allowed for model training
    
    data = remove_empty_or_negative_spectra(data)

    # Option to exclude one isotope (for Outlier analysis)
    excluded_isotope = isotope_outlier
    data_unknown, isotope_unknown, data_known, isotopes_known = split_spectra_one_vs_all(data, all_isotopes, excluded_isotope)
    GlobalVariables.isotopes_tr = isotopes_known  # save only known isotopes to global variable for training

    # Training for dimensionality reduction
    train_dim_model(data_known, save_results=False)

    data_known = apply_dim_model(data_known)
    data_unknown = apply_dim_model(data_unknown)

    # Feature Engineering
    for da in [data_known, data_unknown]:
        da = enrich_dicts_with_information(da)

    x_known = []
    for da in data_known:
        combined_list = da["scores_norm_sorted"] + [da["scores_norm_mean"], da["scores_norm_median"], da['explained_variance'], da['cosine_similarity'], da["scores_norm_abs"]]
        x_known.append(combined_list)

    x_unknown = []
    for da in data_unknown:
        combined_list = da["scores_norm_sorted"] + [da["scores_norm_mean"], da["scores_norm_median"], da['explained_variance'], da['cosine_similarity'], da["scores_norm_abs"]]
        x_unknown.append(combined_list)

    return x_known, x_unknown


def check_column_names_match(x_combined, labels_combined, column_names):
    """
    Validates that the number of rows in `x_combined` and `labels_combined` match and that the combined 
    array has the same number of columns as specified in `column_names`.

    Parameters
    ----------
    x_combined : np.ndarray
        The first array to be combined, typically containing feature data.
    labels_combined : np.ndarray
        The second array to be combined, typically containing labels or additional data columns.
    column_names : list of str
        A list of column names that should match the total number of columns in the horizontally stacked array.

    Raises
    ------
    ValueError
        If `x_combined` and `labels_combined` do not have the same number of rows.
    ValueError
        If the combined array has a different number of columns than the length of `column_names`.

    Notes
    -----
    This function horizontally stacks `x_combined` and `labels_combined`, then verifies the column count
    against `column_names`. If the count does not match, it raises an error to prompt manual adjustment.
    """
    # Check if number of rows match
    if x_combined.shape[0] != labels_combined.shape[0]:
        raise ValueError("x_combined and labels_combined must have the same number of rows")
    
    # Stack the arrays horizontally
    combined = np.hstack((x_combined, labels_combined))
    
    # Check if the number of columns matches the length of column_names
    if combined.shape[1] != len(column_names):
        raise ValueError(f"Mismatch: combined array has {combined.shape[1]} columns, "
                         f"but column_names has {len(column_names)} elements. Adjust column_names accordinly manually!")
    

def fit_logistic_regression_for_outlier_feature(df_combined: pd.DataFrame, feature: str):
    """
    Fits a logistic regression to separate known and unknown data based on a feature, e.g. the cosine similarity. 

    :param df_combined: dataframe of known and unknown data
    :type df_combined: pd.DataFrame
    :param feature: column name in dataframe of the feature to be used to distinguish known and unknown data, e.g. 'cos.sim.'
    :type feature: str
    :return: x and y values and fitted logistic regression curve
    """
    
    # Extract every point
    x_data = df_combined[feature].values
    y_data = df_combined["label"].values

    # Improve initial guess based on the data
    x0_guess = np.median(x_data)  # guess midpoint at the median of x data
    k_guess = -1.  # guess slope (this can be adjusted if needed)

    # Improve initial guess based on the data
    x0_guess = np.median(x_data)  # guess midpoint at the median of x data
    k_guess = -1.  # guess slope (this can be adjusted if needed)

    # Increase the number of maximum function evaluations (maxfev)
    popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=[x0_guess, k_guess], maxfev=5000)
    print(f'Fitting parameters of sigmoid (x0, k): {popt}')

    # Generate a smooth curve for plotting the sigmoid fit
    x_fit = np.linspace(min(x_data), max(x_data), 500)
    y_fit = sigmoid(x_fit, *popt)

    return x_data, y_data, x_fit, y_fit


def sigmoid(x, x0, k):
    """
    Sigmoid function definition

    """
    return 1 / (1 + np.exp(-k * (x - x0)))


def SimplePredict(data, cut):
    """
    Function to compare elements of a 1D numpy array with a threshold (cut).
    
    Parameters:
    data (numpy.ndarray): 1D numpy array containing numeric data
    cut (float): A threshold value between 0 and 1
    
    Returns:
    numpy.ndarray: A 1D numpy array where 0 corresponds to values above the cut, and 1 corresponds to values below or equal to the cut.
    """
    return np.where(data > cut, 0, 1)


def set_isotope_outliers(all_isotopes:List[str]):
    """Set isotope_outliers to the values of all_isotopes

    :param all_isotopes: list of all isotopes
    :type all_isotopes: List[str]
    :return: list of isotopes that should play outlier
    :rtype: List[str]
    """
    isotope_outliers = all_isotopes.copy()  # Copy to avoid modifying the original list
    # Remove 'background' if it exists in the list
    if 'background' in isotope_outliers:
        isotope_outliers.remove('background')
    return isotope_outliers