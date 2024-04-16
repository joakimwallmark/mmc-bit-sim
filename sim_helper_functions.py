import torch

def filter_missing_categories(test_data:torch.Tensor, train_data:torch.Tensor):
    """
    Filter out rows in the 'test_data' tensor that contain values in any column 
    not present in the corresponding column of 'train_data' tensor.

    Parameters
    ----------
    test_data : torch.Tensor
        The test data tensor to be filtered.
    train_data : torch.Tensor
        The training data tensor used as reference for filtering.

    Returns
    -------
    tuple
        A tuple containing the filtered 'test_data' tensor and the count of rows 
        with missing categories.
    """
    missing_categories = torch.zeros_like(test_data, dtype=torch.bool)

    for i in range(test_data.size(1)):  # Iterate over columns
        unique_values = torch.unique(train_data[:, i])
        missing_categories[:, i] = ~torch.isin(test_data[:, i], unique_values)

    missing_rows = torch.any(missing_categories, dim=1)
    filtered_dat = test_data[~missing_rows]
    no_missing = torch.sum(missing_rows).item()

    return filtered_dat, no_missing
