import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import confusion_matrix
from tools.util import *
from tools.tools_model import *

def saveplot(save_plot:bool, name:str, title:str):
    """
    Saves open matplotlib figure in directory dir_saveplot. 

    :param save_plot: True if plot should be saved to folder, False if plot is displayed.
    :type save_plot: bool
    :param name: includes information on isotope and (optionally) detector, e.g. 'Am241_right'
    :type name: str
    :param title: title of the plot
    :type title: str
    """
    
    if save_plot:
        try:
            if 'dir_plots' not in globals():  # Check if dir_plots is defined in the global scope
                raise NameError("The directory for saving plots ('dir_plots') is not defined.")

            # convert 'Am241_right' to 'Am241' but leave 'Am241_Ba133_Co60' unchanged
            isotope = name.rsplit('_', 1)[0] if any(name.endswith(f'_{det}') for det in GlobalVariables.all_detectors) else name

            dir_saveplot = join(dir_plots, isotope)
            if not os.path.exists(dir_saveplot):
                os.makedirs(dir_saveplot)

            savename = join(dir_saveplot, f'{title}_{name}.svg')
            plt.savefig(savename, format='svg', bbox_inches='tight')
            print(f'Saving figure to {savename}')

        except NameError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while saving the plot: {e}")
    else:
        plt.show()
    

def get_color_list(labels:List[str]) -> Tuple[List, Dict[str, float]]:
    """
    Counts n_iso, the number of different (unique) isotopes in labels. 
    Builds dictionary that assigns each isotope to a color and applies this mapping to labels.

    :param labels: list of labels, does not have to be unique (can be a flat or nested list)
    :type labels: List[str]

    :return: color_list: list of color (same length as labels), assigning each label to a color.
            color_dict: dictionary where each (unique) isotope is assigned one color
    :rtype: Tuple[List, Dict[str, float]]
    """

    nested = True if any(isinstance(item, list) for item in labels) else False  # check if labels is a nested list
    if nested:
        distinct_labels = sorted(set(tuple(label) for label in labels))   # create list of isotopes
    else:  # for flat list
        distinct_labels = sorted(set(labels))  # create list of isotopes
    n_iso = len(distinct_labels)
    denomi = n_iso - 1 if n_iso > 1 else n_iso  # denominater
    distinct_colors = [plt.get_cmap('rainbow')(1.*i/denomi) for i in range(n_iso)]  # pick n_iso distinct colors
    color_dict = {distinct_labels[i]: distinct_colors[i] for i in range(n_iso)}
    if nested:
        color_list = [color_dict[tuple(x)] for x in labels]
    else:  # for flat list
        color_list = [color_dict[x] for x in labels]
    return color_list, color_dict


def plot_original_and_rebinned_example_spectrum(filename, data_ori, data_rebinned):
    if 'background' not in filename: 
        data_ori = [x for x in data_ori if x['labels'] != ['background']]
        data_rebinned = [x for x in data_rebinned if x['labels'] != ['background']]
    kk = np.random.choice(len(data_ori))  # pick random spectrum

    int_ori = int(np.sum(np.abs(data_ori[kk]['spectrum'])))
    int_rebinned = int(np.sum(np.abs(data_rebinned[kk]['spectrum'])))

    leg_ori = f'Original {filename} spectrum Nr. {kk} ({int_ori} counts)'
    leg_rebinned = f'Rebinned {filename} spectrum Nr. {kk} ({int_rebinned} counts)'

    plt.figure(figsize=(10, 3))
    plt.plot(data_ori[kk]['spectrum'], alpha=0.5, label=leg_ori)
    plt.plot(data_rebinned[kk]['spectrum'], alpha=0.5, label=leg_rebinned)
    plt.xlabel('Channel')
    plt.ylabel('Counts')
    plt.xlim(0, GlobalVariables.n_channels)
    plt.legend(loc='upper right')
    plt.show()


def plot_mean_spectra_by_isotope_and_detector(iso_list:List[str], zoom_in:bool, save_plots:bool):
    """
    Calculates mean spectra of all isotopes given in names_list, for each detector individually.
    Two subplots per isotope: 1. original mean spectra (with bg), 2. normalized mean spectra (bg subtracted).
    The cosine similarity (measure for similarity between spectra) between the means of different detectors is 
    calculated and displayed in right subplot.
    Only works for single-label spectra or spectra containing one isotope + background. 

    :param iso_list: list of isotopes to be shown, e.g. ['Am241', 'Co60', 'Ir192']
    :type iso_list: List[str]
    :param zoom_in: Option to zoom into the lower end of the spectra
    :type zoom_in: bool
    :param save_plots: Option to save the plots to folder. 
    :type save_plots: bool
    """

    for iso in iso_list:  # iterates over isotopes
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        plt.suptitle('Mean & standard error of preprocessed spectra', fontsize=15)
        means_dict = {}

        # create dictionary to assign colors to different detectors in plots
        _, col_dict = get_color_list(GlobalVariables.all_detectors)  
        for det in GlobalVariables.all_detectors:  # iterates over all detectors (e.g. ['left', 'right', 'simulated'])    

            # mean and standard deviation of all spectra in <name>.npy, isotope and background spectra seperately
            mean_bg, std_bg, mean_iso, std_iso = calc_mean(iso, dir_numpy_ready, [det], norm=False)
            int_iso = np.linalg.norm(mean_iso)
            int_bg = np.linalg.norm(mean_bg)
            if int_iso != 0.:
                means_dict[det] = mean_iso
                x = np.arange(len(mean_iso))

                # left subplot: means (not normalized) with error margins, channels on x axis
                axes[0].plot(mean_iso, c=col_dict[det], label=f'{iso} {det}')
                axes[0].fill_between(x, mean_iso-std_iso, mean_iso+std_iso, color=col_dict[det], alpha=0.2)
                axes[0].set_title(f'Mean spectra of {iso} (original)')
                axes[0].set_xlabel('energy channels')
                axes[0].set_xlim(0, len(mean_iso))  # default
                if zoom_in:
                    axes[0].set_xlim(0, 25)  # option to zoom into lower channels

                # Background subtraction and normalization
                if 'background' in iso or det=='simulated':  # no background subtraction
                    pure_iso = mean_iso / int_iso  
                    std_bg = std_bg / int_bg  
                else:  # measured isotope spectra (containing background)
                    pure_iso = mean_iso - mean_bg  # subtract background from mean
                    pure_iso_int = np.linalg.norm(pure_iso) 
                    pure_iso = pure_iso / pure_iso_int 
                    std_bg = std_bg / pure_iso_int

                # convert channels to energies
                offset = GlobalVariables.std_calib['offset']
                slope = GlobalVariables.std_calib['slope']
                quad = GlobalVariables.std_calib['quad']
                energies = offset + slope * x + quad * x**2
                    
                # right subplot: means (normalized) with background subtracted, energies on x axis
                axes[1].plot(energies, pure_iso, c=col_dict[det], label=f'{iso} {det}')
                axes[1].fill_between(energies, pure_iso-std_bg, pure_iso+std_bg, color=col_dict[det], alpha=0.3)
                axes[1].set_title(f'Mean spectra of {iso} (background subtracted, normalized)')
                axes[1].set_xlabel('energy / keV')
                axes[1].set_xlim(0, np.max(energies))
                if zoom_in:
                    axes[1].set_xlim(0, np.max(energies)/10)  # option to zoom into lower channels

                for ax in axes:
                    ax.legend(loc='upper right')
                    ax.set_ylabel('intensity / a.u.')
                    
            else:   # no spectra found for isotope iso: delete figure
                fig.clf()
                plt.close(fig)
        
        if len(means_dict.keys()) > 1:  # multiple means per isotope / multiple detectors
            dets = list(means_dict.keys())
            means = list(means_dict.values())

            # create list of all combinations between detectors (e.g. left&right, left&simulated, right&simulated)
            combinations = list(itertools.combinations(range(len(dets)), 2))
    
            # loop through all combinations and write cosine similarity in right subplot
            for idx, (i, j) in enumerate(combinations):
                cos_sim = cosine_similarity(means[i], means[j])
                dettxt = f'{dets[i]} & {dets[j]}'
                axes[1].annotate(f'Cos similarity betw. {dettxt} : {cos_sim: .2f}', xy=(0.02, 0.91 - idx * 0.1), xycoords='axes fraction') 

        plt.tight_layout()
        title = f'compare_means_of_different_detectors{"_zoom" if zoom_in else ""}'
        saveplot(save_plots, iso, title)


def plot_example_spectra_by_isotopes(iso_list:List[str], n_spectra, save_plot:bool):
    """
    Plots multiple example spectra (one plot per isotope). 
    Only works for single-label spectra or spectra containing one isotope + background. 

    :param iso_list: list of isotopes to be shown, e.g. ['Am241', 'Co60', 'Ir192']
    :type iso_list: List[str]
    :param n_spectra: 'all' or number of maximum spectra plotted
    :type n_spectra: int or str
    :param save_plot: Option to save plot to folder
    :type save_plot: bool
    """

    _, color_dict = get_color_list(iso_list)  # dictionary to assign colors to different isotopes in plots

    for iso in iso_list:
        data, _ = load_spectral_data(dir_numpy_ready, GlobalVariables.all_detectors, include_files=f'{iso}.npy')
        spectra = [x['spectrum'] for x in data]
        labels = [x['labels'] for x in data]
        if n_spectra != 'all': 
            indices = np.random.choice(np.arange(len(labels)), n_spectra)  # pick n_spectra spectra randomly
            spectra = [spectra[i] for i in indices]
            labels = [labels[i] for i in indices]
        iso_spectra = np.array([spec for spec, lab in zip(spectra, labels) if iso in lab])
        colore = color_dict[iso]  # get color for isotope
        if len(iso_spectra) > 0:
            fig = plt.figure(figsize=(10, 2))
            plt.plot(iso_spectra.T, color=colore)
            plt.xlabel('energy channel', fontsize=16)
            plt.ylabel('counts', fontsize=16)
            plt.xlim(0, len(spectra[0]))

            # set up legend with patches
            patchy = mpatches.Patch(color=colore, label=iso)
            plt.legend(handles=[patchy], loc='upper right', fontsize=14)
            saveplot(save_plot, iso, 'example_spectra')
        else: 
            print(f'No spectra labelled as {iso} found in {iso}.npy, cannot be plotted.')


def plot_cos_sim_matrix_means(cosine_similarity_matrix:np.ndarray, names:List[str], threshold:float):
    """
    Visualizes the cosine similarity as triangular matrix between all isotope spectra means (detectors separately).
    Orange-rimmed: means of same isotope, different detector, that are not similar enough (cos_sim < threshold)
    Red-rimmed: means of different isotopes that are too similar (cos_sim >= threshold)

    Only works for single-label spectra or spectra containing one isotope + background. 

    :param cosine_similarity_matrix: cosine similarity matrix between mean spectra of all isotopes and detectors 
                                    (only lower triangle)
    :type cosine_similarity_matrix: np.ndarray
    :param names: list of xticklabels
    :type names: List[str]
    :param threshold: threshold for cosine similarity (minimum value for means of same isotope, 
                        maximum value for means of different isotopes)
    :type threshold: float
    """
    
    if not -1. <= threshold <= 1.:
        raise ValueError(f'The threshold was set to {threshold} but has to be a number between -1 and 1!')

    n_names = len(names)
    fig, ax = plt.subplots(figsize=(12, 12))

    if cosine_similarity_matrix.shape[0] != cosine_similarity_matrix.shape[1] or cosine_similarity_matrix.shape[0] != n_names:
        raise ValueError("The cosine similarity matrix must be square and match the length of the names list.")

    # mask upper triangle (no background colors)
    mask = np.triu(np.ones_like(cosine_similarity_matrix, dtype=bool), k=1)
    masked_matrix = np.ma.masked_where(mask, cosine_similarity_matrix)
    
    # plot masked cosine similarity matrix
    cax = ax.matshow(masked_matrix, cmap='viridis')
    for i in range(n_names):
        for j in range(n_names):
            cos_sim = cosine_similarity_matrix[i, j]
            if i >= j:  # only lower left triangle
                text = f'{cos_sim:.2f}'
                color = 'white' if cosine_similarity_matrix[i, j] < 0.5 else 'black'
                ax.text(j, i, text, va='center', ha='center', color=color, fontsize=10)  # write value of cosine similarity into each square
                
                # Draw orange box around values of same isotope, different detector that are too low (not similar enough)
                if names[i].split('_')[0]==names[j].split('_')[0] and names[i]!=names[j]:
                    if cos_sim < threshold:  
                        print(f'Not similar enough: {names[i]} and {names[j]}, cosine similarity={cos_sim:.3} in row {i}, column {j}')
                        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fc='none', ec='orange', lw=2, clip_on=False))
                
                # Draw red box around values of different values that are too high (too similar)
                if names[i].split('_')[0]!=names[j].split('_')[0]:
                    if cos_sim >= threshold:
                        print(f'Too similar: {names[i]} and {names[j]}, cosine similarity={cos_sim:.3}, in row {i}, column {j}')
                        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fc='none', ec='darkred', lw=2, clip_on=False))
    ax.set_xticks(np.arange(n_names))
    ax.set_yticks(np.arange(n_names))
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticklabels(names)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=True)
    ax.spines[:].set_visible(False)

    # adjust colorbar
    fig.colorbar(cax, ax=ax, label='Cosine similarity', orientation='vertical', location='left', pad=0.17, shrink=0.7, aspect=50)
    cax.set_clim(0, 1)

    plt.show()


def plot_loadings_subplots(save_plot:bool):
    """
    Visualize the loadings (transformation matrix from spectral into latent space, consists of stacked mean spectra).
    Each mean spectrum of an isotope is plotted in a subplot. In addition, the part of the spectrum considered in model 
    inference (only channels > min_channel_tr) is highlighted in red. 

    :param save_plot: Option to save the plot.
    :type save_plot: bool
    """
    
    # get isotopes, number of components and min_channel
    isotopes = GlobalVariables.isotopes_tr
    n_comp = len(isotopes)
    min_channel = GlobalVariables.min_channel_tr
    
    # normalize loadings
    loadings_norm = normalize_spectra(GlobalVariables.loadings_tr)
    
    # set up figure and define colors
    fig, axes = plt.subplots(n_comp, figsize=(7, 1.2*n_comp))
    x = np.arange(GlobalVariables.n_channels)
    color_list, _ = get_color_list(isotopes)

    for k, iso in enumerate(isotopes):
        
        # plot normalized loadings
        axes[k].plot(loadings_norm[k, :], c='grey', alpha=.8, label=f'PC {k+1}: {iso}')
        
        # plot normalized loadings (only channels > min_channel_tr)
        axes[k].plot(x[min_channel:], loadings_norm[k, min_channel:], linewidth=3, \
                     c=color_list[k], alpha=.8, label=f'channels ≥ {min_channel}')
        
        axes[k].set_xlim(-1, GlobalVariables.n_channels)

        if k < n_comp - 1:  # turn ticks off for all subplots but the last one
            axes[k].tick_params(axis='x', bottom=False, labelbottom=False)
        axes[k].set_ylabel('Intensity', fontsize=12)  # label for y axes 
        axes[k].legend(loc='upper right')

    axes[k].set_xlabel('Energy channel', fontsize=15)  # label for x axis: lowest subplot only

    plt.suptitle('Loadings', fontsize=20)  # set title
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.)

    saveplot(save_plot, '_'.join(isotopes), title='Loadings_subplots')



def plot_confusion_matrix(data_te:List[Dict], class_type:str, save_plot:bool):
    """
    Plots the classification results as confusion matrix (predicted vs. true labels) using seaborn.
    This reveals which labels were confused in model inference. 
    Squares are framed in different colors according to correctness and completeness of the classification: 
    - green frame: complete & correct classification (for single-label and multi-label)
    - golden frame: incomplete, correct classification (only for multi-label)
    - orange frame: complete, partially incorrect classification (only for multi-label)
    - red frame: incomplete, partially incorrect classification (for single-label and multi-label)


    :param data_te: test dataset (list of dictionaries with each dict containing one spectrum and metadata)
    :type data_te: List[Dict]
    :param class_type: can be 'single-label' (use only dominant/first label for each spectrum) or 
    'multi-label' (use all labels)
    :type class_type: str
    :param save_plot: option to save the plot
    :type save_plot: bool
    """
    
    if class_type not in ['single-label', 'multi-label']:
        raise ValueError(f'Invalid class_type: {class_type}. Must be either "single-label" or "multi-label".')


    if class_type == 'single-label':
        labels_true = [x['labels'][0] for x in data_te]  # keep only first label (either isotope or pure background)
        labels_predict = [list(x['labels_pred_dict'].keys())[0] for x in data_te]  # keep only first key of dictionary
    elif class_type == 'multi-label':
        labels_true = [' '.join(sorted(x['labels'])) for x in data_te]  # converts ['Am241', 'background'] to 'Am241 background'
        labels_predict = [' '.join(sorted(list(x['labels_pred_dict'].keys()))) for x in data_te]
    
    # create list of class labels (serve as axes of the confusion matrix)
    class_labels_true = sorted(set(labels_true))  # x axis of confusion matrix
    class_labels_predict = sorted(set(labels_predict))  # y axis of confusion matrix

    # map labels to indices
    label_to_index_true = {label: i for i, label in enumerate(class_labels_true)}  
    label_to_index_predict = {label: i for i, label in enumerate(class_labels_predict)}

    # set up empty confusion matrix
    confusion_matrix = np.zeros((len(class_labels_predict), len(class_labels_true)), dtype=int)

    # fill confusion matrix by looping over lists of predicted and true labels
    for true, pred in zip(labels_true, labels_predict):
        confusion_matrix[label_to_index_predict[pred], label_to_index_true[true]] += 1

    # set up figure & plot confusion matrix as seaborn heatmap
    fig_width = len(class_labels_true) * 0.7
    fig_height = len(class_labels_predict) * 0.5
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': 'counts', 'shrink': 0.5},
                xticklabels=class_labels_true, yticklabels=class_labels_predict)

    # draw frames around the frames with their colors indicating the correctness & completeness of the prediction
    for true, pred in zip(labels_true, labels_predict):
        true_idx = label_to_index_true[true]
        pred_idx = label_to_index_predict[pred]
        
        # green frame: complete & correct classification
        if set(pred) == set(true):
            color = 'green'
        # golden frame: incomplete, correct classification
        elif set(pred).issubset(set(true)):
            color = 'gold'
        # orange frame: complete, partially incorrect classification
        elif set(true).issubset(set(pred)):
            color = 'darkorange'
        # red frame: incomplete, partially incorrect classification 
        elif not set(pred).issubset(set(true)) and not set(true).issubset(set(pred)):
            color = 'darkred'
        else:
            continue  
        ax.add_patch(plt.Rectangle((true_idx, pred_idx), 0.93, 0.93, fill=False, edgecolor=color, linewidth=2))
    plt.xlabel('true labels', fontsize=15)
    plt.ylabel('predicted labels', fontsize=15)
    plt.xticks(rotation=-40, ha='left') 
    plt.title('Confusion Matrix')
    saveplot(save_plot, '', title='Confusion_matrix')


def plot_misclassified_spectra(data_te:List[Dict], class_type:str, save_plot:bool):
    """
    Plots three misclassified spectra, each in one line, in two subplots.
    Left side: In addition to the original spectrum, the mean spectrum of the correct and predicted label 
    are plotted (in green and red) and the scores of the correct and predicted label are printed. 
    Right side: In addition to the original spectrum, the denoised spectrum is plotted and 
    their cosine similarity is calculated.

    :param data_te: test dataset (list of dictionaries with each dict containing one spectrum and metadata)
    :type data_te: List[Dict]
    :param class_type: can be 'single-label' (use only dominant/first label for each spectrum) or 
    'multi-label' (use all labels)
    :type class_type: str
    :param save_plot: option to save the plot
    :type save_plot: bool
    """
    
    if class_type not in ['single-label', 'multi-label']:
        raise ValueError(f'Invalid class_type: {class_type}. Must be either "single-label" or "multi-label".')


    # divide data into correct and erroneous predictions
    data_corr, data_err = identify_misclassifications(data_te, class_type)

    # calculate error ratio over whole test dataset
    err_ratio = len(data_err) / (len(data_corr) + len(data_err))

    # Choose misclassified spectra to be shown 
    n_show = 3
    data_err = np.random.choice(data_err, n_show)

    # set up figure
    fig = plt.figure(figsize=(15, 2.4*n_show))
    plt.suptitle(f'Misclassified spectra ({err_ratio: .2%})')
    
    for i, da in enumerate(data_err):
        spectrum = np.array(da['spectrum'])
        counts = np.sum(spectrum)
        spectrum = spectrum / counts  # normalize spectrum
        
        # extract scores and map to isotopes in a dictionary
        scores = np.array(da['scores_norm'])
        scores_dict = {iso: score for iso, score in zip(GlobalVariables.isotopes_tr, scores)}
        
        if class_type == 'single-label':
            label_true = da['labels'][0]  # extract first true label
            label_pred = list(da['labels_pred_dict'].keys())[0]  # extract dominant predicted label

        elif class_type == 'multi-label':
            label_true = ' '.join(sorted(da['labels']))  # converts ['Am241', 'background'] to 'Am241 background'
            label_pred = ' '.join(sorted(da['labels_pred_dict'].keys()))  

        
        # left side: misclassified spectrum and means of true and predicted labels
        ax_left = fig.add_subplot(n_show, 2, 2*i+1)
        labeltext_left = f'counts: {round(counts)}\n detector: {da["detector"]}\n absorber thickness: {da["absorber thickness"]}'
        ax_left.plot(spectrum, c='black', label=labeltext_left)  # plot misclassified spectrum
        ax_left.set_xlim(0, GlobalVariables.n_channels)

        # set up dictionary to map isotopes to principal components (columns of loadings_tr)
        loadings_dict = {iso: pc for iso, pc in zip(GlobalVariables.isotopes_tr, GlobalVariables.loadings_tr)}

        if class_type == 'single-label':
            # extract & normalize mean of predicted isotope
            pc_predict = loadings_dict[label_pred]  # mean spectrum (predicted isotope)
            pc_predict = normalize_spectra(pc_predict)

            # extract & normalize mean of true isotope
            pc_true = loadings_dict[label_true]  # mean spectrum (true isotope)
            pc_true = normalize_spectra(pc_true)

            # plot both means in left subplot
            ax_left.plot(pc_predict, c='firebrick', alpha=0.6, label=f'predicted: {label_pred}')
            ax_left.plot(pc_true, c='green', alpha=0.6, label=f'correct label: {label_true}')
            
            # annotate predicted and true scores in left subplot
            annot_text_left0 = f'scores predict ({label_pred}): {scores_dict[label_pred]: .3} \n' 
            annot_text_left1 = f'scores true ({label_true}) : {scores_dict[label_true]: .3}'
            annot_text_left = annot_text_left0 + annot_text_left1
            ax_left.annotate(annot_text_left, xy=(0.1, 0.8), xycoords='axes fraction')
        
        elif class_type == 'multi-label':

            for iso in da['labels_pred_dict'].keys():
                # extract & normalize means of predicted isotopes, plot them in left subplot
                pc_predict = loadings_dict[iso] 
                pc_predict = normalize_spectra(pc_predict)
                ax_left.plot(pc_predict, c='firebrick', alpha=0.6, label=f'predicted: {iso}')
                
            for iso in da['labels']:
                if iso in loadings_dict.keys():
                    # extract & normalize means of true isotopes, plot them in left subplot
                    pc_true = loadings_dict[iso]
                    pc_true = normalize_spectra(pc_true)
                    ax_left.plot(pc_true, c='green', alpha=0.6, label=f'correct label: {iso}')
                else: 
                    print(f'ERROR: No principal component found for {iso}. Maybe {iso} was not included in model training?')
            
            # annotate predicted scores 
            print(f'{label_true=}')
            print(f'{da["labels_pred_dict"].keys()}')
            annot_text_right0 = {iso: round(sco, 2) for iso, sco in da['labels_pred_dict'].items()}
            if label_true in da["labels_pred_dict"].keys():
                annot_text_right1 = f'scores true ({label_true}):  {da["labels_pred_dict"][label_true]:.3} '
            else: 
                annot_text_right1 = ''
            
            def wrap_text(text, width):
                return '\n'.join([text[i:i+width] for i in range(0, len(text), width)])

            wrap_width = 45  # maximum width of annotation line
            annot_text_right = (f'scores predict: {wrap_text(str(annot_text_right0), wrap_width)}\n'
            f'{wrap_text(annot_text_right1, wrap_width)}')

            ax_left.annotate(annot_text_right, xy=(0.01, 0.8), xycoords='axes fraction')

        ax_left.set_xlabel('channel')
        ax_left.legend()


        # extract and normalize denoised spectrum
        denoised = np.array(da['denoised'])
        denoised = denoised / np.sum(np.abs(denoised))

        # calculate cosine similarity between origina
        cos_sim = cosine_similarity(denoised, spectrum)  

        # right subplot: original & denoised spectrum
        ax_right = fig.add_subplot(n_show, 2, 2*i+2)
        ax_right.plot(spectrum, c='orange', linewidth=2, label=label_true)
        ax_right.plot(denoised, c='gray', label='denoised spectrum')
        ax_right.set_xlabel('channel')
        ax_right.set_xlim(0, GlobalVariables.n_channels)
        ax_right.annotate(f'cosine similarity: {cos_sim: .3}', xy=(0.1, 0.9), xycoords='axes fraction')
        ax_right.legend()
    
    plt.tight_layout()
    saveplot(save_plot, '_'.join(GlobalVariables.isotopes_tr), title='Misclassified_spectra')


def plot_denoised_spectrum_example(data:List[dict]):
    """
    The original and denoised spectrum are plotted for a random example spectrum and their cosine similarity is printed. 

    :param data: test dataset (list of dictionaries with each dict containing one spectrum and metadata)
    :type data: List[dict]
    """
    # pick a random spectrum and extract original, denoised spectrum, true and predicted labels
    n_spectra = len(data)
    ii = np.random.choice(np.arange(n_spectra))  
    original_spectrum = data[ii]['spectrum']
    denoised_spectrum = data[ii]['denoised']

    labels = data[ii]['labels']
    labels = labels[0] if len(labels) == 1 else labels   # convert ['Am241'] to 'Am241'

    labels_pred_dict = data[ii]['labels_pred_dict']
    labels_pred_dict = {iso: round(sco, 2) for iso, sco in labels_pred_dict.items()}  # round scores

    cos_sim = cosine_similarity(original_spectrum, denoised_spectrum)
    exp_var = explained_variance_score(original_spectrum, denoised_spectrum)
    
    plt.figure(figsize=(12, 3))
    plt.title(f'Denoised spectrum ({labels})')
    plt.plot(original_spectrum, c='darkorange', lw=2., label='original spectrum')
    plt.plot(denoised_spectrum, c='black', lw=1.5, label='denoised spectrum')  
    plt.annotate(f'predicted isotopes: {labels_pred_dict}', xy=(0.02, 0.95), xycoords='axes fraction')
    plt.annotate(f'cosine similarity: {cos_sim: .2}', xy=(0.02, 0.89), xycoords='axes fraction')
    plt.annotate(f'explained variance ratio: {exp_var: .2%}', xy=(0.02, 0.83), xycoords='axes fraction')
    plt.xlim(0, GlobalVariables.n_channels)
    plt.xlabel('energy channels', fontsize=16)
    plt.ylabel('intensity', fontsize=16)
    plt.legend(fontsize=14)
    plt.show()


def plot_misclassification_statistics(data_te:List[Dict], class_type:str):
    """
    Plots histograms of correct (green) and wrong (red) classifications versus different quantities: 

    - explained variance ratio (measure for the explained variance between original & denoised spectrum)
    - cosine similarity (measure for the similarity between original & denoised spectrum)
    - integral of the spectrum (number of counts), e.g. to reveal if spectra with too few counts are more likely to be misclassified

    In addition, a threshold can be adjusted for each subplot. It can serve as a decision boundary in a measurement and 
    classification routine, deciding whether to trust a prediction or not. They can be altered below.
    
    As an example, if the accuracy improves significantly when only predictions with `cosine similarity > 0.85` are considered, 
    this rule can be applied in the measurement routine. 

    :param data_te: test dataset (list of dictionaries with each dict containing one spectrum and metadata)
    :type data_te: List[Dict]
    :param class_type: can be 'single-label' (use only dominant/first label for each spectrum) or 
    'multi-label' (use all labels)
    :type class_type: str
    """
  
    # divide data into correct and erroneous predictions
    data_corr, data_err = identify_misclassifications(data_te, class_type)

    # set up empty arrays for all quantities (they will be filled in the loop below)
    exp_var_corr, exp_var_err = [], []
    exp_var_both = [exp_var_corr, exp_var_err]

    cos_sim_corr, cos_sim_err = [], []
    cos_sim_both = [cos_sim_corr, cos_sim_err]

    abs_corr, abs_err = [], []
    abs_both = [abs_corr, abs_err]

    int_corr, int_err = [], []
    int_both = [int_corr, int_err]

    for i, data in enumerate([data_corr, data_err]):  # iterate over correct/incorrect data
        if not data:  
            continue  # skip loop if list is empty

        # extract original and denoised spectra
        spectra = [x['spectrum'] for x in data]
        denoised_spectra = [x['denoised'] for x in data]

        # calculate explained variance between original and denoised spectra, save to lists
        exp_var_list = []
        for ori, denoised in zip(spectra, denoised_spectra):
            exp_var = calculate_explained_variance(ori, denoised)
            exp_var_list.append(exp_var)
        exp_var_both[i].extend(exp_var_list)

        # calculate cosine similarity between original and denoised spectra, save to lists
        cos_sim_list = [cosine_similarity(a, b) for a, b in zip(spectra, denoised_spectra)]
        cos_sim_both[i].extend(cos_sim_list)

        # calculate length of scores vector, save to lists
        absolute_values = [np.linalg.norm(x['scores_norm']) for x in data]
        abs_both[i].extend(absolute_values)

        # calculate integrals of spectra, save to lists
        integrals = np.sum(np.abs(spectra), axis=1)
        int_both[i].extend(integrals)

    # define thresholds / decision boundaries (can be altered)
    thresh_expvar = 0.95
    thresh_cos = 0.95
    thresh_int = 2500
    thresh_abs = 0.9

    # rearrange lists for plotting
    quantities_corr = [exp_var_corr, cos_sim_corr, abs_corr, int_corr]
    quantities_err = [exp_var_err, cos_sim_err, abs_err, int_err]
    threshs = [thresh_expvar, thresh_cos, thresh_abs, thresh_int]

    # set up figure with 3 subplots
    n_subplots = 4
    fig, axes = plt.subplots(n_subplots, figsize=(18, 3*n_subplots))
    xlabels = ['explained variance ratio between denoised & original spectrum)', 
               'cosine similarity between denoised & original spectrum', 
               'absolute value / length of scores vector', 
               'integral of spectrum (total number of counts)']
    
    # iterate over correct / incorrect predictions
    for i, (q_corr, q_err) in enumerate(zip(quantities_corr, quantities_err)):
        fig2, axes2 = plt.subplots(1)  # set up dummy figure to extract histogram data
        bins = np.linspace(min(q_corr+q_err), max(q_corr+q_err), 200)
        n_corr, bins_corr, _ = axes2.hist(q_corr, bins=bins)  # histogram data for correct predictions
        n_err, bins_err, _ = axes2.hist(q_err, bins=bins)  # histogram data for incorrect predictions
        plt.close(fig2)  # deletes current figure
        
        # set color for bins below and above threshold
        c_corr = ['lightgrey' if y < threshs[i] else 'green' for y in bins_corr[:-1]]
        c_err = ['lightgrey' if y < threshs[i] else 'red' for y in bins_err[:-1]] 

        # calculate number of correct & incorrect predictions above threshold
        n_corr_thresh = len([x for x in q_corr if x > threshs[i]])  
        n_err_thresh = len([x for x in q_err if x > threshs[i]])
        
        if i < 3:  # first three subplots
            axes[i].set_xlim(0, 1)  # cos sim and exp. var. are normalized values
        
        if n_corr_thresh == 0 and n_err_thresh == 0: 
            acc_thresh = 0.

        else: # calculate accuracy in relevant part of histogram (above/below threshold)
            acc_thresh = n_corr_thresh/ (n_corr_thresh + n_err_thresh)  
        ratio_thresh = (n_corr_thresh + n_err_thresh) / len(data_te)

        # plot histogram
        axes[i].bar(bins_corr[:-1], n_corr, width=(bins_corr[1]-bins_corr[0]), color=c_corr, edgecolor='green', alpha=0.8)
        axes[i].bar(bins_err[:-1], -n_err, width=(bins_err[1]-bins_err[0]), color=c_err, edgecolor='red', alpha=0.8)
        axes[i].set_xlabel(xlabels[i], fontsize=18)
        axes[i].set_ylabel('frequency', fontsize=18)

        axes[i].axhline(0, color='black', linewidth=0.8)  # horizontal line at y = 0
        axes[i].axvline(threshs[i], color='black', linewidth=2.)  # vertical line for threshold
        text = f'For threshold {threshs[i]}: \n{ratio_thresh: .1%} of the data can be classified with accuracy {acc_thresh: .1%}'
        axes[i].annotate(text, xy=(0.02, 0.8), xycoords='axes fraction', fontsize=15)
    fig.suptitle('Misclassification statistics', fontsize=30)
    fig.tight_layout()


def plot_scores_scatter_matrix(data_tr:List[Dict], data_te:List[dict], class_type:List[dict], n_dim_max:int, only_errs:bool, save_plot:bool):
    """
    Plots n-dimensional scores as scatter matrix, in each subplot opposing two dimensions / principal components.
    Each training spectrum is displayed as scatter point, colored by its true label. 
    Test data are framed in black and colored by their predicted label. 
    Optionally, only misclassified test data can be displayed (for only_errs = True).

    :param data_tr:  training dataset (list of dictionaries with each dict containing one spectrum and metadata)
    :type data_tr: List[Dict]
    :param data_te: test dataset (list of dictionaries with each dict containing one spectrum and metadata)
    :type data_te: List[dict]
    :param class_type: can be 'single-label' (use only dominant/first label for each spectrum) or 
    'multi-label' (use all labels)
    :type class_type: List[dict]
    :param n_dim_max: maximum number of dimension to be plotted. Default: n_dim = 6
    :type n_dim_max: int
    :param only_errs: option to plot only misclassified test data. Default: only_errs = True
    :type only_errs: bool
    :param save_plot: option to save the plot
    :type save_plot: bool
    """
    
    if only_errs:  # option to show only misclassified test data
        _, data_te = identify_misclassifications(data_te, class_type)
        print('In plot_scores_scatter_matrix, only misclassified test data are displayed.')
    
    # Choose which random training and test data are displayed
    n_spectra_tr = len(data_tr)
    n_spectra_te = len(data_te)
    ind_tr = list(np.random.choice(np.arange(n_spectra_tr), min(n_spectra_tr, 5000)))  # limit displayed training data
    ind_te = list(np.random.choice(np.arange(n_spectra_te), min(n_spectra_te, 300)))  # limit displayed test data
    data_tr = [data_tr[i] for i in ind_tr]
    data_te = [data_te[i] for i in ind_te]

    # extract scores of training and test data
    scores_tr = np.array([x['scores_norm'] for x in data_tr])
    scores_te = np.array([x['scores_norm'] for x in data_te])
    
    # extract labels from training and test data, assign to colors
    labels_tr = np.array([x['labels'][0] for x in data_tr])
    labels_pred = [list(x['labels_pred_dict'].keys())[0] for x in data_te]
    c_tr, color_dict = get_color_list(labels_tr)
    c_pr, _ = get_color_list(labels_pred)
    
    if class_type == 'multi-label':
        print('CAUTION: for the scores scatter matrix, colors only reflect the dominant component of multi-label spectra!')
    
    # determine number of subplots and set up figure
    n_dim_real = scores_tr.shape[1]
    n_rows = n_cols = min(n_dim_real, n_dim_max)  # limit number of displayed principal components
    fig, axes = plt.subplots(n_rows-1, n_cols-1, figsize=(2*(n_rows-1), 2*(n_cols-1)))

    for j in range(1, n_rows):  # iterates over rows (dimensions in latent space)

        for k in range(n_cols-1):  # iterates over columns (dimensions in latent space)

            if j<=k:  # row <= column
                axes[j-1][k].axis('off')  # show only lower left triangle of the figure

            else:  # row > column (lower left triangle): oppose dimension j and dimension k
                
                axes[j-1][k].scatter(scores_tr[:, k], scores_tr[:, j], s=5, color=c_tr, label=labels_tr)  # plot training data
                
                if len(scores_te) > 0:  # plot test data
                    axes[j-1][k].scatter(scores_te[:, k], scores_te[:, j], s=15, color=c_pr, edgecolor='black', linewidth=.5)                
    
                axes[j-1][k].legend().remove()  # remove  legend

            # set ticks and ticklabels of subplots
            ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]

            if j < n_rows-1:  # turn off ticks for all rows but the last one
                axes[j-1][k].set_xticks([])
                axes[j-1][k].tick_params(bottom=False)

            else:  # labels & ticklabels of last row
                axes[j-1][k].set_xlabel(GlobalVariables.isotopes_tr[k], fontsize=15)
                axes[j-1][k].set_xticks(ticks)
                axes[j-1][k].set_xticklabels(ticks, fontsize=8)

            if k > 0:  # turn off ticks for all columns but the first one
                axes[j-1][k].set_yticks([])
                axes[j-1][k].tick_params(left=False)

            else:  # labels & ticklabels of first column
                axes[j-1][k].set_ylabel(GlobalVariables.isotopes_tr[j], fontsize=15)
                axes[j-1][k].set_yticks(ticks)
                axes[j-1][k].set_yticklabels(ticks, fontsize=8)

    # set up legend
    patches = [mpatches.Patch(color=x[1], label=x[0]) for x in color_dict.items()]
    tr_marker = mlines.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=10, label='Training data')
    te_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=12, markerfacecolor='grey', markeredgewidth=1, label='Test data')

    fig.legend(handles = patches + [tr_marker, te_marker], bbox_to_anchor=(1, 0.8), fontsize=15)
    
    plt.suptitle('Spectra in latent space: Scores as scatter matrix', fontsize=20)
    plt.tight_layout() 
    plt.subplots_adjust(wspace=0., hspace=0.)

    # optionally: save plot to folder
    concatname = '_'.join(GlobalVariables.isotopes_tr)  # concatenate isotopes to one string as filename
    saveplot(save_plot, concatname, title='Scores_in_scatter_matrix')



def plot_mean_scores_barplot(data_te:List[Dict], class_type:str, save_plot:bool):
    """
    Plots the mean of the predicted scores of each isotope as bar plot (x axes: isotopes_tr).
    For multi-label data containing more than one isotope are ignored, only the first label is taken into account.
    Only single-label data (not containing background, e.g. simulated spectra) or the combination of one isotope + background are considered. 

    For single-label classification of spectral data not containing background, this should lead to distinct bars, 
    each isotope having values close to 1 for its corresponding isotope axis in latent space. 

    Reveals if the dimensionality reduction succeeds to map single-label spectra mainly to the correct isotope axis 
    in latent space. This helps to identify which isotopes can be clearly distinguished and which may be mistaken.

    :param data_te: test dataset (list of dictionaries with each dict containing one spectrum and metadata)
    :type data_te: List[Dict]
    :param class_type: classification type (can be 'single-label' or 'multi-label')
    :type class_type: str
    :param save_plot: option to save the plot
    :type save_plot: bool
    """
    
    if class_type == 'multi-label':
        print(f'Mean scores cannot be visualized as bar plot for {class_type=}')
    else: 
        # extract scores and true labels of test data, assign colors
        scores_te = np.array([x['scores_norm'] for x in data_te])
        labels_te = [x['labels'][0] for x in data_te]  
        _, color_dict = get_color_list(labels_te)
        
        # set up figure
        fig = plt.figure(figsize=(20, 3))
        n_iso_tr = len(GlobalVariables.isotopes_tr)  # number of isotopes used in model training
        x = np.arange(n_iso_tr)  # set up x axis
        width = 1 / (n_iso_tr + 1)  # calculate width for each x tick
        
        for ll, iso in enumerate(GlobalVariables.isotopes_tr):  # iterates over isotopes used in model training
            # find all scores labelled as iso or iso + background
            ind_iso = [ii for ii, lab in enumerate(labels_te) if iso in lab and set(lab).issubset({iso, 'background'})]
            ind_iso = [i for i, lab in enumerate(labels_te) if lab==iso]
            scores_iso = scores_te[ind_iso, :]
            
            if len(scores_iso) > 0:  # test if any scores are found for isotope iso
                mean_score_iso = np.mean(scores_iso, axis=0)  # calculate mean
                print(f'ratio of mean {iso} scores on {iso} axis: {mean_score_iso[ll]:.2}')  # print score of the correct axis
                
                # normalize mean scores and standard deviation
                int_mean_score = np.sum(np.abs(mean_score_iso))
                mean_score_iso = mean_score_iso / int_mean_score  # normalize mean scores
                std_score_iso = np.std(scores_iso, axis=0) / int_mean_score  # normalize standard deviation
                
                # plot as bar plot, including error bars
                plt.bar(x + ll*width, mean_score_iso, yerr=std_score_iso, width=width, capsize=1.5, \
                        color=color_dict[iso], label=f'{iso} spectra')
                plt.xticks(x, GlobalVariables.isotopes_tr)
                plt.xlabel('axis in latent space, corresponding to isotope', fontsize=15)
                plt.ylabel('share of scores', fontsize=15)
                plt.title(f'Share of scores (mean value, aggregated by true labels)', fontsize=20)
                plt.legend(bbox_to_anchor=(1.13, 1.1), loc='upper right')
            
            else:
                print(f'No spectra found labelled as {iso}')

        concatname = '_'.join(GlobalVariables.isotopes_tr)
        saveplot(save_plot, concatname, title='mean_scores_barplot')


def identify_misclassifications(data_te:List[Dict], class_type:str) -> Tuple[List[Dict], List[Dict]]:
    """
    Divides test data into datasets with correct / incorrect predictions. 
    For class_type = 'single-label', only the first label is considered and background is ignored in labels containing
    one isotope + background. 
    For class_type = 'multi-label', all labels are considered and any difference between true and predicted 
    labels is counted as error.

    :param data_te: test dataset (list of dictionaries with each dict containing one spectrum and metadata)
    :type data_te: List[Dict]
    :param class_type: can be 'single-label' (use only dominant/first label for each spectrum) or 
    'multi-label' (use all labels)
    :type class_type: str

    :return: data_corr: subset of data_te with correct predictions, 
            data_err: subset of data_te with incorrect predictions
    :rtype: Tuple[List[Dict], List[Dict]]
    """

    if class_type not in ['single-label', 'multi-label']:
        raise ValueError(f'Invalid class_type: {class_type}. Must be either "single-label" or "multi-label".')

    # copy data_te (for safe alterations of the variable without affecting it outside of the function)
    data = copy.deepcopy(data_te)
    
    if class_type == 'single-label':

        labels_te = [x['labels'] for x in data]  # extract true labels of test data
        
        if np.any([len(lab) > 2 for lab in labels_te]):  # check if data_te contains multi-label data
            print(f'You chose {class_type=} but your test data contain multi-label spectra!')
        
        # filter out background from isotope spectra (convert ['Am241', 'background'] to ['Am241'])
        labels_te_without_bg = [l if l==['background'] else list(filter(lambda x: x!='background', l)) for l in labels_te]
        
        # overwrite property 'labels' in data with background-filtered labels
        for da, lab in zip(data, labels_te_without_bg):
            da['labels'] = lab
        
        # subdivide data into datasets with correct / incorrect predictions (only first label counts)
        data_corr = [x for x in data if list(x['labels_pred_dict'].keys())[0] == x['labels'][0]]
        data_err = [x for x in data if list(x['labels_pred_dict'].keys())[0] != x['labels'][0]]

    elif class_type == 'multi-label':

        # concatenate list of true and predicted labels for each spectrum: ['Am241', 'Co60'] -> 'Am241 Co60'
        labels_te = [' '.join(sorted(x['labels'])) for x in data]
        labels_pred = [' '.join(sorted(x['labels_pred_dict'].keys())) for x in data]

        # subdivide data into datasets with only-correct / incorrect predictions
        data_corr = [x for x, true, pred in zip(data, labels_te, labels_pred) if pred == true]
        data_err = [x for x, true, pred in zip(data, labels_te, labels_pred) if pred != true]
    
    return data_corr, data_err


def plot_outlier_confusion(test_targets, prediction):
    """
    Plot a confusion matrix to compare actual vs predicted outlier labels.

    Parameters:
    - test_targets (array-like): True labels of test data (0 for known, 1 for unknown).
    - prediction (array-like): Predicted labels from the model (0 for known, 1 for unknown).
    """
    cm = confusion_matrix(test_targets, prediction)

    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=(3, 2.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Known \nisotope', 'Unknown \nisotope'], 
                yticklabels=['Known \nisotope', 'Unknown \nisotope'])
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('Actual Labels', fontsize=12)
    plt.title('Confusion Matrix')
    plt.show()


def plot_feature_importance(feature_names,y):
    """
    Visualize feature importance using a scatter plot.

    Parameters:
    - feature_names (list of str): Names of the features used in the model.
    - y (array-like): Importance scores for each feature.
    """
    fig_featureimportance = plt.figure(figsize=(10, 2))
    plt.scatter(feature_names, y)
    plt.xticks(rotation=90)
    plt.ylabel('feature importance / a.u.')
    plt.show()


def plot_fitted_sigmoid(x_data, y_data, x_fit, y_fit):
    """
    Plot the data points and fitted sigmoid curve for logistic regression on cosine similarity.

    Parameters:
    - x_data (array-like): X-axis data points (cosine similarity scores).
    - y_data (array-like): Y-axis data points (outlier labels).
    - x_fit (array-like): X values for the fitted sigmoid curve.
    - y_fit (array-like): Y values for the fitted sigmoid curve.
    """
    plt.figure(figsize=(12, 3))
    plt.scatter(x_data, y_data, label='Data Points', color='b', alpha=0.1)

    # Plot the fitted sigmoid curve (bounded between 0 and 1)
    plt.plot(x_fit, y_fit, label='Fitted Sigmoid Curve', color='r')
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Outlier Score")

    plt.show()


def plot_metrics_vs_threshold(thresholds, accuracies, precisions, recalls):
    """
    Plot accuracy, precision, and recall scores as a function of cosine similarity threshold.

    Parameters:
    - cuts (array-like): Threshold values for cosine similarity.
    - accuracies (array-like): Accuracy scores for each threshold.
    - precisions (array-like): Precision scores for each threshold.
    - recalls (array-like): Recall scores for each threshold.
    """
    plt.figure(figsize=(12, 3))
    plt.plot(thresholds, accuracies, label="Accuracy") 
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")

    # Adding labels and title
    plt.xlabel('threshold for cosine similarity')
    plt.ylabel('Score')
    plt.title('Choose the desired threshold (here for 10% Outliers in training)')
    plt.xlim(np.min(thresholds), np.max(thresholds))
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
