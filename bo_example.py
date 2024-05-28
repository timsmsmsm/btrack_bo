# Import the necessary functions from bo_example_funcs.py
from src.optimization import *
from src.ground_truth_trees import *
from src.predicted_trees import *

# other imports
import ctctools
import os
import urllib.request
import zipfile
from traccuracy.loaders import load_ctc_data

if __name__ == '__main__':
    # Select a dataset to download from the Cell Tracking Challenge website
    dataset_name = "Fluo-N2DH-SIM+"
    url = f"http://data.celltrackingchallenge.net/training-datasets/{dataset_name}.zip"
    data_dir = 'downloads'
    file_path = os.path.join(data_dir, f"{dataset_name}.zip")

    # Create download directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # Download the dataset if it's not already downloaded
    if not os.path.exists(file_path):
        print(f"Downloading {dataset_name} data from the CTC website")
        urllib.request.urlretrieve(url, file_path)
        
        # Unzip the data
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    voxel_sizes = (0.645, 0.645, 1.0) # Voxel size in microns, write third value as 1.0 if data is 2D
    datasets_path = data_dir
    scale = compute_scaling_factors(voxel_sizes) # Compute scaling factors to make dataset isotropic

    # Load the ground truth data
    gt_data = load_ctc_data(
        f'{datasets_path}/{dataset_name}/01_GT/TRA',
        f'{datasets_path}/{dataset_name}/01_GT/TRA/man_track.txt',
        name=f'{dataset_name}_GT'
    )

    # Load the dataset
    dataset = ctctools.load(f'{datasets_path}/{dataset_name}', experiment="01", scale=scale)

    # Choose the sampler you want to use
    sampler = 'tpe' # Choose from ['random', 'tpe', 'NSGA-II', 'cmaes']

    #1obj: AOGM only optimisation, 2obj: AOGM and MBC optimisation
    objectives = '1obj' # Choose from ['1obj', '2obj']

    #study name as it will be saved in sqlite:///btrack.db
    study_name = 'example_study_1'

    n_trials = 32 #number of trials
    timeout = 120 #seconds
    use_parallel_backend = True #use parallel backend for faster optimization

    #for optimisation without timeout/pruning use optimize_dataset() function instead
    study = optimize_dataset_with_timeout(dataset, gt_data, objectives, study_name, n_trials=n_trials, timeout=timeout, use_parallel_backend=use_parallel_backend, sampler=sampler)

    for trial in study.best_trials:
        write_best_params_to_config(trial.params, f'{dataset_name}_{trial.number}.json') #write best parameters to json file



    #PLOT GROUND TRUTH LINEAGES:
    output_path = "gt_trees" # Directory to save the output image

    # Create directory if it doesn't exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    process_and_plot_ground_truth(data_dir, output_path, dataset_name)



    #PLOT PREDICTED LINEAGES:
    file_path = os.path.join(data_dir, f"{dataset_name}")
    trial = study.best_trials[0]
    config_file_path = f'{dataset_name}_{trial.number}.json' #write best parameters to json file
    output_path = "pred_trees"  # Specify the output path for the image

    # Create directory if it doesn't exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    voxel_sizes = (0.125, 0.125, 1.0)  # Voxel size in microns, write third value as 1.0 if data is 2D
    scale = compute_scaling_factors(voxel_sizes)  # Scale can be specified here
    load_from_dict_flag = False  # Specify if the config should be loaded from dict or JSON file

    conf = initialize_config(config_file_path=config_file_path, load_from_dict_flag=load_from_dict_flag)
    data, properties, graph, dataset = load_and_configure_tracker(file_path, conf, scale)
    plot_lineage_tree(data, properties, graph, dataset, dataset_name, output_path)