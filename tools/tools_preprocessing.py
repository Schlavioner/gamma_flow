from .globals import GlobalVariables
from .util import *
from .plotting import plot_original_and_rebinned_example_spectrum


def check_folder_structure():
    """
    Ensures that the essential folders for data input (spectral data) and output (plots, trained models) are built (relative paths).

    In addition, .txt file that define the isotopes that should be preprocessed are necessary of dir_data. It should contain a list of clear
    isotope names, e.g. Am241, Eu152, multi-class, background (where multi-class stands for spectra that include multiple isotopes at once).

    The isotopes defined in the .txt file should be a substring in all spectral data files of this isotope so they can be concatenated
    to one file in concatenate_isotope_datasets().
    Example: The file '20240805_Am241_highcountrates.npy' will be matched to the isotope 'Am241'.

    Files of isotopes that are missing in the .txt file will not be preprocessed!
    If the .txt file is missing, it is generated automatically but still has to be filled in manually.
    """
    essential_folders = [
        dir_plots,
        dir_trained_models,
        dir_data,
        dir_numpy_raw,
        dir_numpy_ready,
    ]

    # Builds essential folders if they are not present yet
    for fol in essential_folders:
        if not os.path.exists(fol):
            print(f"Building folder {fol}")
            os.makedirs(fol)
    print("All essential folders should now be available.")

    # define path to the essential txt file
    isolist_path = join(dir_data, "00_list_of_isotopes")

    if not os.path.isfile(isolist_path):
        with open(isolist_path, "w") as file:
            line1 = "You need to manually list the isotopes of your training/test data here. \n"
            line2 = "Please do not include other text or data in this file. \n"
            line3 = "Example: Am241, Ba133, Co60, multi-class, background..."
            file.write(line1 + line2 + line3)
        raise FileExistsError(f"""The txt file with the list of isotopes ({isolist_path}) is missing. 
        It was now created but you need to manually list the measured isotopes there, 
        e.g. Am241, Co60, Cs137, background """)
    else:
        print(f"The file {isolist_path} exists.")
        file = open(isolist_path, "r")
        content = file.read()
        if "You need to manually list the isotopes" in content:
            raise NameError(f"""It seems like you have not manually edited the txt-file {isolist_path} yet. 
            It should only include the comma-seperated list of your measured isotopes.
            e.g. Am241, Co60, Cs137, background """)
        else:
            print(
                f"Seems like you have successfully edited the txt-file {isolist_path}. Well done! \n"
            )


def clear_preprocessed_data(paths):
    """
    Deletes the content (previously preprocessed data) of the specified paths.
    Otherwise, analysis scripts might unintentionally use old/outdated data.

    :param paths: relative paths to the folders that should be cleared.
    :type paths: list of strings
    """

    for folder_path in paths:
        if not os.path.exists(folder_path):  # checks if folder exists
            print(f"The folder {folder_path} does not exist.")
            return
        print(f"Deleting content of folder {folder_path}.")
        for filename in os.listdir(folder_path):  # iterates through folder content
            file_path = os.path.join(folder_path, filename)  # builds path name
            try:
                if os.path.isfile(file_path) or os.path.islink(
                    file_path
                ):  # element is a file
                    os.remove(file_path)
                elif os.path.isdir(file_path):  # element is a folder
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error when trying to delete {file_path}. Reason: {e}")


def read_isotope_list() -> List[str]:
    """
    Opens the (manually written) txt-file of the measured/simulated isotopes
    Purpose: users can specify in this txt file which isotopes to include in preprocessing and set filenames.

    :return: list of the isotopes as defined in the txt files in dir_data, e.g. ['Am241', 'Eu152', 'background']
    :rtype: List[str]
    """

    rawtext = open(join(dir_data, "00_list_of_isotopes")).readlines()[0]
    rawtext = ", ".join(rawtext.splitlines())  # remove line breaks

    list_of_isotopes = rawtext.split(",")  # convert to list
    list_of_isotopes = [x.replace(" ", "") for x in list_of_isotopes]  # remove blancs
    list_of_isotopes = [
        x.replace("\n", "") for x in list_of_isotopes
    ]  # remove line breaks
    list_of_isotopes = sorted(
        list(dict.fromkeys(list_of_isotopes))
    )  # remove duplicates, sort alphabetically

    return list_of_isotopes


def rebinning(
    all_files_list: List[str], std_calib: Dict, show_plot: bool
) -> Dict[str, List[Dict]]:
    """
    Reads spectral data and (original) energy calibrations from folder numpy_raw.
    Rebinning: Conversion from channels to energies (using the original energy calibration), then
    conversion from energies back to standardized channels (using the given standard calibration)

    :param all_files_list: list of all files (Output of create_list_of_spectra_files())
    :type all_files_list: List[str]
    :param std_calib: standard (target) energy calibration parameters, e.g. std_calib = {'offset': -10.3, 'slope': 8.1, 'quad': 0.001}
    :type std_calib: Dict
    :param show_plot: option to plot an original and rebinned example spectrum vs. the energy channels
    :type show_plot: bool

    :return: dictionary with filenames as keys and data (list of dicts) as value where each dict corresponds to one spectrum + metadata.
        e.g.    {'Am241_1_right':   [{'spectrum': [...], 'labels':[...]}, {'spectrum': ..., 'labels'}],
                '240805_Ba133_left': [{'spectrum': [...], 'labels':[...]}, {'spectrum': ..., 'labels'}]}
    :rtype: Dict[str, List[Dict]]
    """

    print(" Starting rebinning of spectra...")
    dict_of_data = {}  # set up dictionary for data {filename: data} with data: list of dicts
    for filename in all_files_list:  # iterates over isotopes
        data_ori = np.load(join(dir_numpy_raw, filename), allow_pickle=True)
        data_rebinned = copy.deepcopy(data_ori)
        spectra = np.array([dicty["spectrum"] for dicty in data_ori])  # extract spectra
        n_channels = spectra.shape[1]  # extract number of channels
        GlobalVariables.n_channels = n_channels  # save to global variable
        i = np.arange(n_channels)

        # calculate energies corresponding to the detector channels (using the given standard calibration)
        std_energies = (
            std_calib["offset"] + std_calib["slope"] * i + std_calib["quad"] * i**2
        )

        for jj, (dicty_ori, dicty_rebinned) in enumerate(
            zip(data_ori, data_rebinned)
        ):  # iterates over spectra
            ori_calib = dicty_ori["calibration"]  # original energy calibration

            # calculate energies corresponding to the detector channels (original energy calibration)
            ori_energy = (
                ori_calib["offset"] + ori_calib["slope"] * i + ori_calib["quad"] * i**2
            )  # original energ

            # define interpolation function
            interp_func = interp1d(
                ori_energy,
                spectra[jj, :],
                kind="linear",
                bounds_error=False,
                fill_value=0,
                axis=0,
            )
            rebinned_spectrum = interp_func(std_energies)
            rebinned_spectrum[rebinned_spectrum < 0] = 1e-6  # replace negative values

            dicty_rebinned["spectrum"] = (
                rebinned_spectrum  # update spectrum in dictionary
            )
            dicty_rebinned["calibration"] = (
                std_calib  # update energy calibration in dictionary
            )

        data_rebinned = [
            x for x in data_rebinned if not np.any(np.isnan(x["spectrum"]))
        ]  # check for NaNs
        dict_of_data[filename] = (
            data_rebinned  # list of dicts with filename and data (data is a list of dicts)
        )

        if show_plot:  # plot original and rebinned spectrum
            plot_original_and_rebinned_example_spectrum(
                filename, data_ori, data_rebinned
            )
    return dict_of_data


def concatenate_isotope_datasets(
    dict_of_data: Dict[str, List[Dict]],
) -> Dict[str, List[Dict]]:
    """
    Checks there are multiple datasets per isotope & detector in dict_of_data.
    If so, they are concatenated. If not, they are passed on.
    Example: 'Co60_1_right' and 'Co60_2_right' are concatenated to 'Co60_right'.

    :param dict_of_data: dictionary in the form {filename: data} where data is a list of dictionaries
            e.g.    {'Am241_1_right':   [{'spectrum': [...], 'labels':[...]}, {'spectrum': ..., 'labels'}],
                    '240805_Ba133_left': [{'spectrum': [...], 'labels':[...]}, {'spectrum': ..., 'labels'}]}
    :type dict_of_data: Dict[str, List[Dict]]

    :return: dictionary in the form {isotope_detector: data} where data is a list of dictionaries
         e.g.   {'Am241_right': [{'spectrum': [...], 'labels':[...]}, {'spectrum': ..., 'labels'}],
                'Ba133_left':   [{'spectrum': [...], 'labels':[...]}, {'spectrum': ..., 'labels'}]}
    :rtype: Dict[str, List[Dict]]
    """

    print(
        "\n Starting to concatenate files of same isotope (leave detectors separate)..."
    )
    filename_list = dict_of_data.keys()  # defines which files exist in dict_of_data

    dict_of_data_concat1 = {}  # set up dictionary for concatenated data {isotope_detector: data}
    bg_synonyms = [
        "background",
        "Background",
        "Untergrund",
        "untergrund",
        "Hintergrund",
        "hintergrund",
    ]
    for isotope in (
        GlobalVariables.all_isotopes
    ):  # iterate over isotopes (keywords by which datasets are concatenated)
        for det in GlobalVariables.all_detectors:  # iterate over detectors
            new_name = f"{isotope}_{det}"

            # find all files for this isotope and detector
            if isotope == "background":
                iso_files = [
                    key
                    for key in filename_list
                    if any(syn in key for syn in bg_synonyms) and det in key
                ]
            else:
                iso_files = [
                    key for key in filename_list if isotope in key and det in key
                ]

            if len(iso_files) > 1:  # more than one file per isotope and detector
                print(f'The files {iso_files} are concatenated to "{isotope}_{det}".')

                # Concatenate all data from same isotope and same detector (e.g. 'Co60_1_right' and 'Co60_2_right')
                data_concat1 = [da for name in iso_files for da in dict_of_data[name]]

                # write to dict_of_data with isotope and detector as key (e.g. 'Co60_right')
                dict_of_data_concat1[new_name] = data_concat1

            elif len(iso_files) == 1:  # only one file per isotope and detector
                data_extracted = dict_of_data[
                    iso_files[0]
                ]  # extract data from dict_of_data
                dict_of_data_concat1[new_name] = (
                    data_extracted  #  write to dict_of_data_concat1
                )
    return dict_of_data_concat1


def concatenate_detector_datasets(
    dict_of_data: Dict[str, List[Dict]],
) -> Dict[str, List[Dict]]:
    """
    Checks isotope-wise if there are multiple datasets from different detectors.
    Concatenates data of different detectors (e.g. right, left, simulated) to one dataset
    Example: 'Co60_left', 'Co60_right' and 'Co60_sim' are concatenated and saved as 'Co60.npy'

    :param dict_of_data: dictionary in the form {isotope_detector: data} where data is a list of dictionaries
         e.g.   {'Am241_right': [{'spectrum': [...], 'labels':[...]}, {'spectrum': ..., 'labels'}],
                'Ba133_left':   [{'spectrum': [...], 'labels':[...]}, {'spectrum': ..., 'labels'}]}
    :type dict_of_data: Dict[str, List[Dict]]

    :return: dictionary in the form {isotope: data} where data is a list of dictionaries
         e.g.   {'Am241': [{'spectrum': [...], 'labels':[...]}, {'spectrum': ..., 'labels'}],
                'Ba133':   [{'spectrum': [...], 'labels':[...]}, {'spectrum': ..., 'labels'}]}
    :rtype: Dict[str, List[Dict]]
    """

    print("\n Starting to concatenate files of same isotope for all detectors...")

    filename_list = sorted(
        dict_of_data.keys()
    )  # defines which files exist in dict_of_data
    dict_of_data_concat2 = {}  # set up dictionary for concatenated data {isotope: data}
    for isotope in (
        GlobalVariables.all_isotopes
    ):  # iterate over isotopes (keywords by which datasets are concatenated)
        # find all files for this isotope
        iso_files = [key for key in filename_list if key.split("_")[0] == isotope]

        if len(iso_files) > 1:  # more than one file per isotope
            print(f'The datasets {iso_files} are concatenated to "{isotope}".')

            # Concatenate all data from same isotope (different detectors), e.g. 'Co60_left' and 'Co60_right')
            data_concat2 = [dicty for name in iso_files for dicty in dict_of_data[name]]

            # write to dict_of_data with isotope as key (e.g. 'Co60')
            dict_of_data_concat2[isotope] = data_concat2

        elif len(iso_files) == 1:  # only one file per isotope
            data_array = dict_of_data[iso_files[0]]  #  extract data from dict_of_data
            dict_of_data_concat2[isotope] = data_array  # write to dict_of_data_concat2
        else:
            print(f"No file found for isotope {isotope}.")

    return dict_of_data_concat2


def limit_spectra_per_isotope(dict_of_data: Dict[str, List[Dict]], n_max: int = 10000):
    """
    Limits the number of spactra per isotope to n_max.
    Saves one .npy-file per isotope in directory dir_numpy_ready.

    :param dict_of_data: dictionary in the form {isotope: data} where data is a list of dictionaries
         e.g.   {'Am241': [{'spectrum': [...], 'labels':[...]}, {'spectrum': ..., 'labels'}],
                'Ba133':   [{'spectrum': [...], 'labels':[...]}, {'spectrum': ..., 'labels'}]}
    :type dict_of_data: Dict[str, List[Dict]]
    :param n_max: maximum number of spectra per isotope, defaults to 10000
    :type n_max: int, optional
    """

    if n_max is None:  # number of spectra is not limited
        print(f"\n You chose not to limit the number of spectra per dataset.")

    name_list = sorted(dict_of_data.keys())  # get unique list of keys (isotopes)
    for name in name_list:
        data = dict_of_data[name]  # extract data
        print(f"{name}: {len(data)} spectra originally")
        if n_max is not None:
            data = limit_length_of_dataset(data, n_max)
            print(f"Now limiting number of spectra per file to {n_max}.\n")
        save_list_of_dicts(dir_numpy_ready, data, name)


def calc_cos_sim_matrix_means(iso_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Calculates cosine similarity between all mean spectra for all isotopes (different detectors individually).
    Results in symmetric matrix, only the lower left triangle is kept.
    Only works for single-label spectra or spectra containing one isotope + background.

    The result can be passed to plot_cos_sim_matrix for visualization.

    :param iso_list: list of isotopes to be shown, e.g. ['Am241', 'Co60', 'Ir192']
    :type iso_list: List[str]

    :return: cosine similarity matrix between mean spectra of all isotopes and detectors (only lower triangle)
    :rtype: np.ndarray
    """

    all_data, _ = load_spectral_data(dir_numpy_ready, GlobalVariables.all_detectors)
    all_means_dict = {}  # set up dictionary of type {isotope_detector: mean_spectrum}
    for iso in iso_list:
        data_iso = [d for d in all_data if iso in d["labels"]]
        det_iso = sorted(
            set([d["detector"] for d in data_iso])
        )  # unique list of detectors
        for det in det_iso:
            name = f"{iso}_{det}"
            _, _, mean_iso, _ = calc_mean(iso, dir_numpy_ready, [det], norm=False)
            all_means_dict[name] = mean_iso
        if len(det_iso) == 0:
            print(
                f"No spectra labelled as {iso} found in {iso}.npy, cannot be included in the cosine similarity matrix! \n"
            )

    # set up cosine similarity matrix
    names = list(all_means_dict.keys())
    n_names = len(names)
    cosine_similarity_matrix = np.zeros((n_names, n_names))

    for i, j in itertools.product(range(n_names), repeat=2):
        mean_el_i = all_means_dict[names[i]]
        mean_el_j = all_means_dict[names[j]]
        cossi = cosine_similarity(mean_el_i, mean_el_j)  # calculate cosine similarity
        cosine_similarity_matrix[i, j] = cossi

    return cosine_similarity_matrix, names
