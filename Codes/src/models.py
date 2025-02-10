import numpy as np
import pandas as pd


from assertions import (
    assert_type,
    assert_metadata_is_valid,
    assert_list_of_type,
    assert_true,
    assert_columns_not_in_dataframe,
)
from misc import (
    get_column_names_with_role,
    get_subset_by_role,
)


def encode_categorical_label_for_classification(
    y, metadata_y, one_hot_encode=False
):
    """Parse and convert the target variable for classification tasks.

    In binary- and multi-classification tasks,
    the target variable *y* is categorical.
    This method will examine the target dataframe to obtain
    the ground truth labels for the target variable *y*
    by examining the metadata for the column with role ``'ground truth'``,
    convert them into a 0-based integer encoding vector,
    and return the encoding and the corresponding set of unique labels.
    It verifies that none of the ground truth labels
    are in fact column names in ``y``.
    *NaN* values for the ground truth label will be passed through
    as a *NaN* in the encoding for the corresponding entries in *y*,
    and will *not* be included in the
    set of unique labels returned; i.e. *NaN* is not a label.
    Multi-label classification is not supported yet.

    Parameters
    ----------
    y : pandas.DataFrame
        The dataframe containing the target variable.
        Exactly one column must have role ``'ground truth'``.
    metadata_y : pandas.DataFrame
        Metadata for ``y``.
    one_hot_encode : bool, default is False
        Whether or not labels should be one-hot encoded. If True, a 2-D
        array of one-hot encoded labels is returned. If False, label encoding
        is applied.

    Returns
    -------
    tuple
        A tuple of length 2 with entries:

        - 0: If one_hot_encode is False, a dataframe that is identical to
             ``y``, except that the single column with role ``'ground truth'``
              has been replaced by an integer-encoded version of itself, where
            the values of the encoding are integers in
            *[0, ..., <number of distinct labels>]*. If one_hot_encode is True,
            a DataFrame of one hot encoded columns is returned (see pandas
            get_dummies documentation:
          https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)
        - 1: a list of strings giving the unique labels
            in the ground truth column of ``y``.
            the index of each label within the list
            corresponds to the integer encoding in the column with
            role ``'ground truth'`` in the dataframe
            in the first entry of this output tuple, described above.
    """

    # check
    assert_metadata_is_valid(metadata_y, y)

    # get the column name which has the ground truth labels for the target.
    ground_truth_column = get_column_names_with_role(
        y, metadata_y, "ground truth"
    )
    assert_list_of_type(ground_truth_column, str)
    if len(ground_truth_column) != 1:
        msg = (
            "Multi-label classification is not yet supported for "
            "the Temporal CNN model for "
            "forecasting classification, but the "
            f"``y`` contained {len(ground_truth_column)} "
            "columns with role 'ground truth'.\n"
            "ground_truth_columns = {ground_truth_column}\n"
            f"y:\n{y.__str__()}"
        )
        raise NotImplementedError(msg)

    ground_truth_column = ground_truth_column[0]

    # get the unique set of labels.
    # Note: *NaN* stays as *NaN*.
    unique_labels = (
        y[ground_truth_column].astype("category").cat.categories.tolist()
    )

    # check
    if len(unique_labels) < 2:
        msg = (
            f"Only {len(unique_labels)} distinct found, but at least 2 "
            + "distinct labels are necessary for classification.\n"
            + "unique_labels = {unique_labels}\n"
        )
        raise AssertionError(msg)

    labels_in_columns_of_y = [
        z for z in unique_labels if z in y.columns.tolist()
    ]
    if len(labels_in_columns_of_y) > 0:
        msg = (
            "Some of the categorical labels in the ground truth for the "
            + "target variable in ``y`` overlap with the names of "
            + "columns in the dataframe ``y``. "
            + "This is unsafe; it could end up overwriting "
            + "columns of ``y``. Choose different column names "
            + "in ``y`` or different categorical labels for the target "
            + "variable.\n"
            + f"unique_labels = {unique_labels}\n"
            + f"columns of y = {y.columns.tolist()}\n"
        )
        raise AssertionError(msg)

    if one_hot_encode:
        assert_columns_not_in_dataframe(y, unique_labels)

        y_encoded = y.copy()
        y_encoded[unique_labels] = pd.get_dummies(
            y_encoded[ground_truth_column]
        )

    else:
        # encode labels as integers. ``np.nan`` gets converted to ``-1``.
        category_type = pd.api.types.CategoricalDtype(categories=unique_labels)
        encoded_ground_truth = (
            y[ground_truth_column].astype(category_type).cat.codes
        )

        # represent *NaN* as ``np.nan`` rather than ``-1``.
        encoded_ground_truth[encoded_ground_truth == -1] = np.nan

        # build a version of ``y`` where the ground truth column is
        # replaced by the integer encoded version of itself.
        y_encoded = y.copy()
        y_encoded[ground_truth_column] = encoded_ground_truth

    return y_encoded, unique_labels


def get_all_torch_optimizer_names():
    """Get a list of the names of available optimizers in Torch.

    Returns
    -------
    list
        A list of strings, giving the names of the optimizers
        available in ``torch``.
    """

    # imports
    import inspect
    import torch

    # get all optimizers in ``torch``
    all_optimizers = inspect.getmembers(torch.optim)
    # list of tuples, where each tuple is of length 2
    # and has entries:
    #  0: the name (string) of the object;
    #  1: the object.

    optimizer_names = [z[0] for z in all_optimizers]
    assert_list_of_type(optimizer_names, str)

    return optimizer_names


def get_torch_optimizer(name):
    """Retrieve the optimizer object from torch by name.

    The optimizer for models is specified by a name (string).
    This function will get the corresponding optimizer object.

    Parameters
    ----------
    name : str
        The name of the requested optimizer object;
        e.g. 'Adam', 'SGD'.

    Returns
    -------
    torch.Optimizer
        The requested optimizer *object* from ``torch``.
    """

    # import
    import inspect
    import torch

    # check
    assert_type(name, str)

    # get all optimizers in ``torch``
    all_optimizers = inspect.getmembers(torch.optim)
    # list of tuples, where each tuple is of length 2
    # and has entries:
    #  0: the name (string) of the object;
    #  1: the object.

    # find the requested optimizer
    req = [z for z in all_optimizers if z[0] == name]
    if len(req) == 0:
        msg = "Unable to find the requested optimizer in the set of "
        msg = msg + "``torch`` optimizers.\n"
        msg = msg + f"requested optimizer = {name}\n"
        msg = (
            msg + f"available optimizers = {get_all_torch_optimizer_names()}\n"
        )
        raise AssertionError(msg)

    assert_true(
        len(req) == 1,
        msg="Utils Error: more than one requested optimizer retrieved",
    )

    # get the optimizer object
    optimizer = req[0][1]

    return optimizer


def get_weights_from_X_or_y(X, metadata_X, y=None, metadata_y=None):
    """Retrieve samplewise weights from the data.

    Samplewise weights are used during model training
    and scoring model performance with metrics.
    Samplewise weights can exist as a single column in ``X`` or ``y``
    or neither, but not both.
    Samplewise weights are indicated by the role ``'weights'`` in
    the metadata.

    This method will examine ``X`` and ``y`` for the existence
    of a column with role ``'weights'``.
    The method verifies that there is only one column with
    role ``'weights'``,
    and that the weights are positive.
    If there are no columns with role ``'weights'``, then it
    returns ``None``.

    Parameters
    ----------
    X : pandas.DataFrame
        The data to make predictions from; i.e. it contains
        at least features.
    metadata_X : pandas.DataFrame
        metadata for ``X``.
    y : pandas.DataFrame | None
        The data containing the target ground truth.
    metadata_y : pandas.DataFrame | None
        metadata for ``y``.

    Returns
    -------
    numpy.ndarray | None
        An array of samplewise weights; length the same as the
        number of rows of ``y`` or ``X``, depending in which
        one it was found.
        Or ``None`` if there is no column in ``X`` nor ``y`
        with role ``'weights'`` in the respective metadata.
    """

    # check
    assert_metadata_is_valid(metadata_X, X)

    if y is not None:
        assert_metadata_is_valid(metadata_y, y)

    if y is not None:
        # weights must be in ``X`` or ``y`` or neither, but not both.
        if ("weights" in metadata_y["role"].tolist()) and (
            "weights" in metadata_X["role"].tolist()
        ):
            msg = "There are columns with role 'weights' in both "
            msg = msg + "``X`` and ``y``. This is ambiguous; "
            msg = msg + "at most one of ``X`` or ``y`` should provide "
            msg = msg + "samplewise weights."
            raise AssertionError(msg)

    # infer weights columns from the roles in the metadata.
    weights_df = None
    if y is not None:
        if "weights" in metadata_y["role"].tolist():
            weights_df = get_subset_by_role(y, metadata_y, "weights")

    if weights_df is None:
        if "weights" in metadata_X["role"].tolist():
            weights_df = get_subset_by_role(X, metadata_X, "weights")

    weights_np = None
    if weights_df is not None:
        # must be only 1 weights column..
        if weights_df.shape[1] > 1:
            msg = "At most one column can provide samplewise weights, "
            msg = msg + f"but {weights_df.shape[1]} columns of "
            msg = msg + "samplewise weights were provided."
            raise AssertionError(msg)

        # get the weights column as a ``pd.Series``.
        weights_np = weights_df.iloc[:0].to_numpy()

    if weights_np is not None:
        # verify that all the weights are positive.
        if np.all((weights_np > 0)) is False:
            bad_arr = weights_np[(weights_np <= 0)]
            msg = "Some of the weights are non-positive.\n"
            msg = msg + "Non-positive weights:\n"
            msg = msg + f"{bad_arr.__str__()}"
            raise AssertionError(msg)

    return weights_np
