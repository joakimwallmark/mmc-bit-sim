# Run as below for MMC and NR simulations respectively (learning rates and batch sizes are based on cross-validation)
# python 2_fit_models.py --model "MMC" --device "cpu" --learning_rates 0.08 --batch_sizes 64
# python 2_fit_models.py --model "NR" --device "cpu" --learning_rates 0.08 --batch_sizes 128
import os
import argparse
import time
import itertools
import datetime
import torch
import torch.multiprocessing as mp
import pandas as pd
from pyarrow import feather
from irtorch import IRT
from irtorch.models import MonotoneNN, NominalResponse
from sim_helper_functions import filter_missing_categories

def parse_args():
    parser = argparse.ArgumentParser(description="Run simulations under different configurations. For list arguments, simulations will be ran using all combinations of the provided values.")
    parser.add_argument('--model', type=str, default='MMC', choices=['MMC', 'NR'], help='Model to use for the simulation (default: MMC).')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to run the simulation on (default: cpu).')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations to run. Note that the maximum is 1000 unless 1_sample_datasets.py is changed. (default: 1000).')
    parser.add_argument('--sample_sizes', type=int, nargs='+', default=[1000, 3000, 5000, 10000], help='Sample sizes (default: [1000, 3000, 5000, 10000]). Provide a space-separated list for multiple values.')
    parser.add_argument('--items', type=int, nargs='+', default=[20, 40, 80], help='Number of items (default: [20, 40, 80]). Provide a space-separated list for multiple values.')
    parser.add_argument('--z_estimation_methods', type=float, nargs='+', default=['ML', 'NN'], help="Method for latent trait estimation (default: ['ML', 'NN']). Provide a space-separated list for multiple values.")
    parser.add_argument('--learning_rates', type=float, nargs='+', default=[0.08], help='Learning rates for training (default: [0.08]). Provide a space-separated list for multiple values.')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[64], help='Batch sizes for training (default: [64]). Provide a space-separated list for multiple values.')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[1], help='Number of MMC hidden layers. Provide a space-separated list for multiple values.')
    return parser.parse_args()

TEST_NAME = "swesat22b_nominal_quant"
full_data = feather.read_feather(f"real_datasets/{TEST_NAME}.feather")
full_data.dropna(inplace=True)
full_data.reset_index(drop=True, inplace=True)
full_data = torch.from_numpy(full_data.to_numpy()).float()
MODEL_MISSING = False
full_data = full_data - 1 # don't model missing

with open(f"real_datasets/item_cat_{TEST_NAME}.txt", "r") as file:
    n_cats = file.read().replace("\n", "")
n_cats = [int(num) for num in n_cats]
with open(f"real_datasets/{TEST_NAME}_correct.txt", "r") as file:
    correct_cat = file.read().replace("\n", "")
correct_cat = [int(num) for num in correct_cat] # don't model missing

def run_simulation(arguments):
    device = arguments['device']
    if device == "cpu":
        torch.set_num_threads(1)  # One thread per core, to avoid overloading the CPU
    error = None
    i = arguments['iteration']
    model_name = arguments['model']
    z_estimation_method = arguments['z_estimation_method']
    n = arguments['n']
    items = arguments['items']
    learning_rate = arguments['learning_rate']
    batch_size = arguments['batch_size']
    hidden_layers = arguments['hidden_layers']
    print(f"Running for i:{i}, n:{n}, items:{items}\n--------------------------")
    try:
        sampled_items = feather.read_feather(f"simulated_datasets/sampled_items_{TEST_NAME}_items{items}.feather")["items"]
        sampled_rows = feather.read_feather(f"simulated_datasets/sampled_rows_{TEST_NAME}_n{n}_items{items}.feather")[i]
        all_rows = set(range(full_data.size(0)))
        excluded_rows = list(all_rows - set(sampled_rows))
        sampled_data = full_data[sampled_rows][:, sampled_items]
        remaining_data = full_data[excluded_rows][:, sampled_items]
        n_cats_task = [n_cats[i]+1 for i in sampled_items]
        correct_task = [correct_cat[i] for i in sampled_items]
        # If some categories are missing in the training data, remove those rows from the test data
        # This makes it more comparable to mirt.
        remaining_data, missing = filter_missing_categories(remaining_data, sampled_data)
    except Exception as e:
        missing = 0
        error = f"Error in reading files for i:{i}, n:{n}, items:{items}: {e}"

    torch.manual_seed(i + 135) # Same seed depending on iteration
    try:
        if "NR" in model_name:
            model_spec = NominalResponse(
                latent_variables=1,
                item_categories=n_cats_task,
                model_missing=MODEL_MISSING,
                mc_correct=correct_task,
                reference_category=False
            )
        elif "MMC" in model_name:
            model_spec = MonotoneNN(
                latent_variables=1,
                item_categories=n_cats_task,
                hidden_dim=[3] * hidden_layers,
                model_missing=MODEL_MISSING,
                mc_correct=correct_task,
                separate="categories",
                negative_latent_variable_item_relationships=True,
                use_bounded_activation=True,
            )

        model = IRT(
            model = model_spec,
            one_hot_encoded=True
        )

        model_save_path = f"fitted_models/{TEST_NAME}_{model_name}_i{i}_n{n}_items{items}_batch_size{batch_size}_learning_rate{learning_rate}_hidden_layers{hidden_layers}_model_missing{int(MODEL_MISSING)}.pt"
        if os.path.exists(model_save_path):
            model.load_model(model_save_path)
            execution_time = None
        else:
            start_time = time.time()
            model.fit(
                train_data=sampled_data,
                validation_data=remaining_data,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
            )
            model.save_model(model_save_path)
            execution_time = time.time() - start_time
    except Exception as e:
        error = f"Error in fitting model for i:{i}, n:{n}, items:{items}: {e}"

    ml_test_save_path = f"ml_scores/ml_test_{TEST_NAME}_n{n}_{items}items_{model_name}_i{i}_batch_size{batch_size}_learning_rate{learning_rate}_hidden_layers{hidden_layers}_model_missing{int(MODEL_MISSING)}.pt"
    if os.path.exists(ml_test_save_path) and z_estimation_method == "ML":
        z_test = torch.load(ml_test_save_path)
    else:
        z_test = model.latent_scores(remaining_data, scale = "z", z_estimation_method=z_estimation_method, ml_map_device=device, lbfgs_learning_rate=0.22)
        if z_estimation_method == "ML":
            torch.save(z_test, ml_test_save_path)

    test_ll = model.log_likelihood(remaining_data, z=z_test, reduction="sum")
    test_residuals = model.residuals(data = remaining_data, z=z_test, average_over="all")

    if error is not None:
        print(f"error: {error}")
    result = {
        "model": model_name,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time": execution_time,
        "iteration": i,
        "n": n,
        "items": items,
        "z_estimation_method": z_estimation_method,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "hidden_layers": hidden_layers,
        "test_log_likelihood": test_ll.item() / remaining_data.size(0) / items,
        "test_residuals": test_residuals.mean().item(),
        "test_obs_with_missing_cats_in_train_data": missing,
        "error": error,
    }

    return result

if __name__ == "__main__":  # Important for multiprocessing in Windows
    args = parse_args()

    MODEL = args.model
    ITERATIONS = args.iterations
    DEVICE = args.device
    SAMPLE_SIZES = args.sample_sizes
    ITEMS = args.items
    Z_ESTIMATION_METHODS = args.z_estimation_methods
    BATCH_SIZES = args.batch_sizes
    LEARNING_RATES = args.learning_rates
    HIDDEN_LAYERS = args.hidden_layers
    iterations = range(args.iterations)

    tasks = [
        {
            "device": DEVICE,
            "model": MODEL,
            "iteration": iteration,
            "z_estimation_method": z_estimation_method,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n": n,
            "items": items,
            "hidden_layers": hidden_layers,
        }
        for iteration, n, items, z_estimation_method,
            batch_size, learning_rate, hidden_layers in itertools.product(
                iterations, SAMPLE_SIZES, ITEMS, Z_ESTIMATION_METHODS,
                BATCH_SIZES, LEARNING_RATES, HIDDEN_LAYERS
        )
    ]

    start_time = time.time()
    if DEVICE == "cpu":
        mp.set_start_method('spawn')
        cores_to_use = mp.cpu_count()
        print(f"Using {cores_to_use} cores")
        with mp.Pool(cores_to_use) as pool:
            results = pool.map(run_simulation, tasks)

        results_df = pd.DataFrame(results)
    elif DEVICE == "cuda":
        result_list = []
        start_time = time.time()
        for params in tasks:
            task_results = run_simulation(params)
            result_list.append(task_results)

        results_df = pd.DataFrame(result_list)

    print(f"Total time: {(time.time() - start_time)/60/60} hours")

    results_df['error'] = results_df['error'].astype('string')
    if results_df["hidden_layers"].dtype != 'int64':
        results_df['hidden_layers'] = pd.to_numeric(results_df['hidden_layers'], errors='coerce')

    file_path = f"simulation_results/py_{TEST_NAME}.feather"
    if os.path.exists(file_path):
        existing_df = feather.read_feather(file_path)
        # Columns to check for complete duplicates
        duplicate_columns = [
            "model", "iteration", "n", "items",
            "z_estimation_method", "learning_rate",
            "batch_size", "hidden_layers", "error"]
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
        results_df = results_df.drop_duplicates(subset=duplicate_columns, keep='last')

    feather.write_feather(results_df, file_path)
