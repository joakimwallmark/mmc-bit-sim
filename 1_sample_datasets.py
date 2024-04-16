import pandas as pd
import numpy as np
from pyarrow import feather

iterations = 1000
n_vec = [1000, 3000, 5000, 10000]
items_vec = [80, 40, 20]
test_name = "swesat22b_nominal_quant"

# Read data
real_data = feather.read_feather(f"real_datasets/{test_name}.feather")
# Remove missing data
real_data.dropna(inplace=True)
real_data.reset_index(drop=True, inplace=True)
n_cats = [int(real_data.iloc[:, col].max()) for col in range(real_data.shape[1])]

np.random.seed(135)
remaining_items = np.arange(80)
for items in items_vec:
    remaining_items = np.random.choice(remaining_items, items, replace=False)
    remaining_items.sort()
    item_df = pd.DataFrame(remaining_items, columns=['items'])
    feather.write_feather(item_df, f"simulated_datasets/sampled_items_{test_name}_items{items}.feather")
    for n in n_vec:
        row_df = pd.DataFrame(index=range(n), columns=range(iterations))
        for i in range(iterations):
            row_df.iloc[:, i] = np.random.choice(range(len(real_data)), n, replace=False)

        feather.write_feather(row_df, f"simulated_datasets/sampled_rows_{test_name}_n{n}_items{items}.feather")
