import numpy as np
import pandas as pd
import misc as fmisc


def assert_values_in_dataframe_column(df, column, values, msg=None):
    """Assert the presence of specific values in a DataFrame column.

    This helper function verifies that all of the
    values specified in `values` are present in
    a column in a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame from pandas.
    column : str
    values : list
        A list of values. If any elements of the list
        are not in the indicated column in the DataFrame,
        then an error is thrown.
    msg : str
        A custom error message. If None,
        a message will be automatically generated.

    Raises
    ------
    AssertionError

    Returns
    -------
    bool
        True.
    """

    assert_type(df, pd.DataFrame, msg=msg)
    assert_type(column, str, msg=msg)
    assert_type_or_list_of_type(values, str, msg=msg)
    values = fmisc.maybe_make_list(values)

    assert_columns_in_dataframe(df, column, msg=msg)

    # gather any values not present in the column.
    absent_vals = [z for z in values if z not in df[column].tolist()]
    if len(absent_vals) > 0:
        jms = ", ".join(absent_vals)
        default_msg = (
            f"Column {column} of the DataFrame lacks the following values: "
            + f"{jms}."
        )
        raise AssertionError(default_msg + (msg or ""))

    return True


def assert_columns_in_dataframe(df, column, msg=None):
    """Assert that columns are present in a DataFrame.

    This helper function verifies that all of the
    specified strings in `columns` are column names
    in a DataFrame. Otherwise, an AssertionError is thrown.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame from pandas.
    column : str or list of str
        Names of column(s) that should be in the DataFrame `df`.
        If any of the strings in `column` are not among
        the column names of `df`, then an AssertionError
        will be thrown.
    msg : str
        A custom error message. If None,
        a message will be automatically generated.

    Raises
    ------
    AssertionError

    Returns
    -------
    bool
        True.
    """

    assert_type(df, pd.DataFrame, msg=msg)
    assert_type_str_or_list_of_str(column, msg=msg)

    if isinstance(column, str):
        if column not in df.columns.tolist():
            default_msg = (
                f"The DataFrame lacks the following column: {column}."
            )
            raise AssertionError(default_msg + (msg or ""))
    else:
        for z in df.columns.tolist():
            assert_columns_in_dataframe(df, z, msg=msg)

    return True


def assert_true(x, msg=None):
    """Assertion wrapper.

    This helper function verifies that a statement is True.
    Otherwise, it throws an Assertion Error.

    Parameters
    ----------
    x : bool
        A boolean statement.
    msg : str
        A custom error message. If None,
        a message will be automatically generated.
    """

    assert_type(x, bool)

    if msg is None:
        msg = "Assertion does not hold."

    if x is False:
        raise AssertionError(msg)

    return True


def assert_metadata_is_valid(metadata, data=None, msg=None):
    """Verify that a dataframe represents metadata.

    This function verifies that a dataframe which is
    purported to be the metadata for some actual
    data really does satisfy the properties to be
    a metadata dataframe.

    In particular, it verifies:

    - The metadata has at least the columns:
        variable,dtype,role.
    - The "special roles" are mutually exclusive;
        i.e. any given column has at most one of
        the special roles.
    - There are no duplicate roles per column.
    - If the corresponding data is provided,
        it checks that all variables in the metadata
        appear in the data, and vice-versa.

    In particular, it verifies column names
    and column types.

    Parameters
    ----------
    metadata : pandas.DataFrame
        It will be verified that this dataframe
        really does represent metadata for some
        data ``data``.
    data : pandas.DataFrame | None
        If ``None``, then the checks on ``metadata``
        will be restricted to those that do not
        rely on the corresponding data.
        Otherwise, the information in ``metadata``
        will be checked against the actual
        columns in ``data``.
    msg : str
        A custom error message. If None,
        a message will be automatically generated.

    Raises
    ------
    AssertionError

    Returns
    -------
    True
    """

    # check
    assert_type(metadata, pd.DataFrame, msg=msg)
    assert_type(data, pd.DataFrame, allow_None=True, msg=msg)

    # check required columns of metadata.
    assert_columns_in_dataframe(
        metadata, ["role", "variable", "dtype"], msg=msg
    )

    # The 'role' and 'variable' columns must contain strings.
    [assert_type(z, str, msg=msg) for z in metadata["role"].tolist()]
    [assert_type(z, str, msg=msg) for z in metadata["variable"].tolist()]

    # no duplicate roles per column
    non_dup_df = metadata.drop_duplicates(subset=["variable", "role"])
    if non_dup_df.shape[0] != metadata.shape[0]:
        default_msg = (
            "The metadata has duplicate entries for column:role pairs.\n"
            + f"Here is the full dataframe:\n{metadata.__str__()}"
        )
        raise AssertionError(default_msg + (msg or ""))

    # Special roles are mutually exclusive:
    # a column can have multiple roles, but it can only have at most
    # one role from the set of special roles.
    special_roles = [
        "identifier",
        "feature",
        "annotation",
        "weight",
        "ground truth",
        "model output",
        "prediction",
    ]

    special_df = metadata[metadata["role"].isin(special_roles)]
    if any(special_df["variable"].duplicated()):
        default_msg = "A variable is allowed to have at most one of the "
        default_msg = (
            default_msg + "special roles, but some of the variables have "
        )
        default_msg = (
            default_msg + "multiple special roles. Consider duplicating the "
        )
        default_msg = (
            default_msg + "column so that each can serve a different role.\n"
        )
        default_msg = default_msg + f"The special roles are:\n{special_roles}."
        raise AssertionError(default_msg + (msg or ""))

    # some of the special roles are mutually exclusive
    # even across variables in the same dataframe.
    contains_feature = any(metadata["role"] == "feature")
    contains_ground_truth = any(metadata["role"] == "ground truth")
    contains_prediction = any(metadata["role"] == "prediction")
    contains_model_output = any(metadata["role"] == "model output")

    if contains_feature and (
        contains_ground_truth or contains_prediction or contains_model_output
    ):
        default_msg = "Some variables have role 'feature', while others "
        default_msg = (
            default_msg + "have role 'ground truth' or 'model output' or "
        )
        default_msg = default_msg + "'prediction', but these are not "
        default_msg = default_msg + "allowed to coexist in the same dataframe."
        raise AssertionError(default_msg + (msg or ""))

    # check against the associated data.
    if data is not None:
        # verify that all columns referred to in ``metadata``
        # are indeed in the data.
        extra_columns = [
            v
            for v in set(metadata["variable"].tolist())
            if v not in set(data.columns.tolist())
        ]
        if len(extra_columns) > 0:
            default_msg = "The metadata is invalid: it refers to "
            default_msg = (
                default_msg + "columns that are not in the associated "
            )
            default_msg = default_msg + "data.\nThe extra columns are:\n"
            default_msg = default_msg + f"{extra_columns}."
            raise AssertionError(default_msg + (msg or ""))

        # verify that all columns in the ``data``
        # are indeed in the ``metadata``.
        missing_columns = [
            v
            for v in set(data.columns.tolist())
            if v not in set(metadata["variable"].tolist())
        ]
        if len(missing_columns) > 0:
            default_msg = "The metadata is invalid: there are columns "
            default_msg = default_msg + "in the data that are not represented "
            default_msg = (
                default_msg + "in the metadata.\nThe missing columns are:\n"
            )
            default_msg = default_msg + f"{missing_columns}."
            raise AssertionError(default_msg + (msg or ""))

    return True


def assert_non_negative_timedelta(t, msg=None):
    """Verify that a timedelta is non-negative.

    Given a timedelta object or a string ``t`` that can be converted to
    a ``pandas.Timedelta``, this function verifies
    that the string represents a non-negative amount of time.

    Parameters
    ----------
    t : str | pandas.Timedelta
        A timedelta object or a string which is compatible with
        ``pandas.Timedelta``.
    msg : str
        A custom error message. If None,
        a message will be automatically generated.

    Raises
    ------
    AssertionError

    Returns
    -------
    bool
        ``True`` if ``t`` represents an amount
        of time that is non-negative.
    """

    # check
    assert_type(t, [str, pd.Timedelta], msg=msg)

    # convert
    if isinstance(t, str):
        td = pd.to_timedelta(t)
    else:
        td = t

    # must be non-negative.
    ts = td.total_seconds()

    if ts < 0:
        default_msg = f"The timedelta {t} is negative. It must be positive."
        raise AssertionError(default_msg + (msg or ""))

    return True


def assert_type(x, t, allow_None=False, msg=None):
    """Assert the type of an object.

    This helper function verifies that an object
    has the indicated type. If not, it will throw
    a TypeError.

    Parameters
    ----------
    x : anything
        Any object whose type you want to verify.
    t : type | list of type
        A 'type' object indicating the type that `x` is required to be,
        or a list or 'type' objects indicating the set of
        permissible types for `x`.
    allow_None : bool
        If True, then no error will be thrown if `x` is None.
    msg : str
        A custom error message. If None,
        a message will be automatically generated.
    """

    # t must be a type or list of types
    if isinstance(t, list):
        # check that all of the objects in the list are types.
        non_types_in_t = [
            type(z).__name__ for z in t if type(z).__name__ != "type"
        ]
        if len(non_types_in_t) > 0:
            jms = ", ".join(non_types_in_t)
            default_msg = (
                "`t` must be an object of type 'type' or a list of types, "
            )
            default_msg = (
                default_msg
                + f"but it contains invalid objects of types: {jms}."
            )
            raise TypeError(default_msg + (msg or ""))

    elif isinstance(t, type) is False:
        default_msg = (
            "`t` must be an object of type 'type' or a list of types, "
        )
        default_msg = default_msg + f"but it is of type: {type(t).__name__}."
        raise TypeError(default_msg + (msg or ""))

    # allow_None must be a bool
    if isinstance(allow_None, bool) is False:
        default_msg = f"""`allow_None` must be of type bool, but is \
            of type '{type(allow_None).__name__}'."""
        raise TypeError(default_msg + (msg or ""))

    if allow_None is True and x is None:
        return True

    # verify the type of `x`.
    if isinstance(t, list) is False:
        t = [t]

    tn_x = type(x).__name__
    permissible_types = [z.__name__ for z in t]

    if tn_x not in permissible_types:
        jms = ", ".join(permissible_types)
        default_msg = f"The variable is of type '{tn_x}', "
        default_msg = default_msg + f"but the permissible types are: {jms}."
        raise TypeError(default_msg + (msg or ""))

    return True


def assert_type_str_or_list_of_str(x, msg=None):
    """Assert that an object is a string or list of strings

    This helper function verifies that an object
    is either a string or a list of strings.
    If not, it will throw a TypeError.

    Parameters
    ----------
    x : anything
        Any object.
    msg : str
        A custom error message. If None,
        a message will be automatically generated.
    """

    ret = assert_type_or_list_of_type(x=x, t=str, allow_None=False, msg=msg)

    return ret


def assert_type_or_list_of_type(x, t, allow_None=False, msg=None):
    """Assert that an object is a certain type or list of those types.

    This helper function verifies that an object
    is either of type ``t`` or is a list of
    objects of type ``t``.
    If not, it will throw a ``TypeError``.

    Parameters
    ----------
    x : Any
        Any object.
    t : type
        A ``type`` object indicating the type that ``x`` is required to be,
        or a list or ``type`` objects indicating the set of
        permissible types for ``x``.
    allow_None : bool
        if ``True``, then also allows for ``x`` to be ``None``.
    msg : str
        A custom error message. If None,
        a message will be automatically generated.

    Returns
    -------
    bool
        ``True`` if ``x`` is of type ``t` or a list of objects
        of type ``t``, or ``None`` and ``allow_None = True``.
    """

    assert_type(allow_None, bool, msg=msg)

    if x is None and allow_None is True:
        return True
    elif isinstance(x, list):
        [assert_type(x=y, t=t, msg=msg) for y in x]
    elif isinstance(x, t):
        return True
    else:
        tn_x = type(x).__name__
        default_msg = (
            f"Must be of type `{t}` or a list of `{t}`, "
            + f"but is of type `{tn_x}`."
        )
        raise TypeError(default_msg + (msg or ""))

    return True


def assert_values(x, values, allow_None=False, msg=None):
    """Assert that a variable takes a permissible value.

    This helper function verifies that the value
    of a variable is permissible, defined as being
    from a pre-defined list of permissible values.

    Parameters
    ----------
    x : anything
        Any object whose value will be verified.
    values : list
        A list of permissible values for `x`.
    allow_None : bool
        If True, then no error will be thrown if `x` is `None`.
    msg : str
        An optional error message to be printed
        if `x` is not in `values`.
        A reasonably informative message is printed by default.
    """

    assert_type(values, list, msg=msg)
    assert_type(allow_None, bool, msg=msg)
    assert_type(msg, str, allow_None=True, msg=msg)

    if allow_None is True:
        if x is None:
            return True

    if x not in values:
        if msg is None:
            msg = f"Impermissible value observed: {x}."
            if len(values) < 20:
                msg = msg + "\nAllowable values are: {*values,}"

        raise ValueError(msg)

    return True


def assert_unique_identifiability_of_rows(df, columns, msg=None):
    """Assert that each row can be uniquely identified.

    This method verifies that each row in a dataframe
    is uniquely identifiable using a given set of columns
    in the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Any dataframe.
    columns : str | list[str]
        The columns that ostensibly uniquely identify
        the rows of ``df``.
    msg : str
        A custom error message. If None,
        a message will be automatically generated.

    Raises
    ------
    AssertionError

    Returns
    -------
    bool
        ``True`` if each row is uniquely identifiable
        by the values in the ``columns``.
    """

    assert_type(df, pd.DataFrame, msg=msg)
    assert_type_or_list_of_type(columns, str, msg=msg)
    assert_columns_in_dataframe(df, columns, msg=msg)

    is_duplicated = df.duplicated(subset=columns)

    if is_duplicated.any() is True:
        default_msg = "Rows are not uniquely identifiable by "
        default_msg = default_msg + f"columns {columns}. In particular, "
        default_msg = (
            default_msg + f"{sum(is_duplicated)} rows are duplicates "
        )
        default_msg = default_msg + "of other rows."
        raise AssertionError(default_msg + (msg or ""))

    return True


def assert_columns_not_in_dataframe(df, columns, msg=None):
    """Assert that columns are *not* present in a DataFrame.

    This helper function verifies that none of the
    specified strings in `columns` are column names
    in a DataFrame. Otherwise, an AssertionError is thrown.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame from pandas.
    columns : str or list of str
        Names of columns that should *not* be in the DataFrame `df`.
        If any of the strings in `columns` are among
        the column names of `df`, then an AssertionError
        will be thrown.
    msg : str
        A custom error message. If None,
        a message will be automatically generated.

    Raises
    ------
    AssertionError

    Returns
    -------
    bool
        True.
    """

    # check
    assert_type(df, pd.DataFrame, msg=msg)
    assert_type_str_or_list_of_str(columns, msg=msg)

    if isinstance(columns, str):
        if columns in df.columns.tolist():
            default_msg = f"The DataFrame already contains column: {columns}."
            raise AssertionError(default_msg + (msg or ""))
    else:
        [assert_columns_not_in_dataframe(df, z, msg=msg) for z in columns]

    return True


def assert_row_count_not_changed(before, after, msg=None):
    """Assert that the number of rows in the data has not changed.

    This helper method verifies that the number of rows
    in the data after the application of an operation
    is the same as the number of rows in the data
    before the operation.
    It does not actually apply the operation.

    Parameters
    ----------
    before : pandas.DataFrame | int | float
        The data before an operation was applied,
        or the number of rows in the data before
        the operation was applied.
    after : pandas.DataFrame | int | float
        The data after an operation was applied,
        or the number of rows in the data after
        the operation was applied.
    msg : str
        A custom error message. If None,
        a message will be automatically generated.

    Raises
    ------
    AssertionError
        If the number of rows before vs after is not the same.

    Returns
    -------
    True
    """

    # check
    assert_type(before, [pd.DataFrame, int, float], msg=msg)
    assert_type(after, [pd.DataFrame, int, float], msg=msg)

    # get number of rows for ``before``:
    if isinstance(before, pd.DataFrame):
        before = before.shape[0]
    elif isinstance(before, float):
        before = fmisc.try_to_cast_to_integer(before)

    assert_true(
        before >= 0,
        msg="Cannot compare row count of a dataframe with negative rows"
        + (msg or ""),
    )

    # get number of rows for ``after``:
    if isinstance(after, pd.DataFrame):
        after = after.shape[0]
    elif isinstance(after, float):
        after = fmisc.try_to_cast_to_integer(after)

    assert_true(
        after >= 0,
        msg="Cannot compare row count of a dataframe with negative rows"
        + (msg or ""),
    )

    # row count must be the same
    if before != after:
        default_msg = "The number of rows in the data was changed by the "
        default_msg = (
            default_msg + "operation, but it was not to supposed to.\n"
        )
        default_msg = default_msg + f"Number of rows before: {before}\n"
        default_msg = default_msg + f"Number of rows after: {after}"
        raise AssertionError(default_msg + (msg or ""))

    return True


def assert_columns_are_numeric(data, columns=None, msg=None):
    """Throw an error is certain columns are not of numeric type.

    This helper function will throw an error if the indicated
    columns of the dataframe are not of a numeric type;
    i.e. ``float``, ``int``, or ``bool``.
    The workhorse is `pandas.api.types.is_numeric_dtype``.

    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe.
    columns : str | list of str | None
        Columns in the ``data`` which should
        be of a numeric type. Numeric type means
        ``float``, ``int``, and ``bool`` .
        If ``None``, then will test all columns.
    msg : str
        A custom error message. If None,
        a message will be automatically generated.

    Raises
    ------
    TypeError

    Returns
    -------
    True
    """

    # imports
    from pandas.api.types import is_numeric_dtype

    # check
    assert_type(data, pd.DataFrame, msg=msg)

    if columns is None:
        columns = data.columns.tolist()

    assert_type_or_list_of_type(columns, str, msg=msg)
    assert_columns_in_dataframe(data, columns, msg=msg)

    # prepare
    columns = fmisc.maybe_make_list(columns)

    # verify column types.
    for colx in columns:
        if is_numeric_dtype(data[colx]) is False:
            default_msg = (
                f"Column '{colx}' of the data must be "
                "a numeric type, but it is of type: "
                "'{data[colx].dtype}'."
            )
            raise AssertionError(default_msg + (msg or ""))

    return True


def assert_list_of_type(x, t, allow_None=False, msg=None):
    """Assert that the object is a list of objects of a given type.

    This helper function verifies that an object
    is a list of objects which have the indicated type.
    If not, it will throw a TypeError.

    Parameters
    ----------
    x : anything
        An object which should be a list where each element
        of the list is one of the types in ``t``.
        All objects in the list need not be of the same type,
        but they must be one of the types in ``t``.
    t : type | list of type
        A ``type`` object indicating the type that ``x`` is required to be,
        or a list or ``type`` objects indicating the set of
        permissible types for ``x``.
    allow_None : bool
        If ``True``, then no error will be thrown if ``x`` is ``None``.
    """

    assert_type(x, list, allow_None=allow_None, msg=msg)
    assert_type_or_list_of_type(x, t, allow_None=allow_None, msg=msg)

    return True


def assert_no_missing_values(data, msg=None):
    """Throw an error if any element of a dataframe is missing.

    This helper function verifies that none of the values
    in any row or column of a dataframe are missing.
    The workhorse is ``pandas.isnull()``.

    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe which will be verified to not
        have any missing values anywhere, in any row
        or column.
    msg : str
        A custom error message. If None,
        a message will be automatically generated.
    """
    # check
    assert_type(data, pd.DataFrame, msg=msg)

    # test
    has_missing_values = bool(data.isnull().values.any())
    assert_type(has_missing_values, bool, msg=msg)

    if has_missing_values is True:
        # get subset of dataframe with missing values
        sub_df = data[data.isnull().any(axis=1)]

        default_msg = f"The dataframe has {sub_df.shape[0]} rows with "
        default_msg = default_msg + "missing values, but it is not allowed to "
        default_msg = default_msg + "have any.\nRows with missing data:\n"
        default_msg = default_msg + f"{sub_df.__str__()}"
        raise AssertionError(default_msg + (msg or ""))

    return True


def assert_columns_have_data(df, columns, msg=None):
    """Assert that the columns in the dataframe are NOT completely null.

    Primarily used to validate the usage of an imputation method

    Parameters
    ----------
    df : pandas.DataFrame
        Data to assess
    columns : str | list of str
        columns in the data to evaluate for full missingness

    Raises
    ------
    AssertionError
        If any of the columns provided are completely missing

    Returns
    -------
    True
    """

    assert_type(df, pd.DataFrame)
    assert_type_str_or_list_of_str(columns)
    columns = fmisc.maybe_make_list(columns)

    if df[columns].isnull().all().any():
        default_msg = "One of the following columns is completely null: "
        default_msg = default_msg + f"{columns}"
        raise AssertionError(default_msg + (msg or ""))

    return True


def assert_identical_roles(metadata1, metadata2, columns1, columns2):
    """Verify that variables have identical roles between two datasets.

    This helper function verifies that columns within a pair of
    corresponding columns between two datasets have identical roles
    in the metadata.

    For example,
    when joining data from two recipes, the metadata of the columns
    which serve as the join keys for the join operation must have
    identical roles in the metadata.

    Specifically, given a list of join keys for the data from the
    left recipe, and a corresponding list of the same length for the
    data from the right recipe, the set of roles in the metadata
    for each corresponding pair of columns (one from left, one from right)
    must be identical.

    Parameters
    ----------
    metadata1 : pandas.DataFrame
        The metadata for the first dataset.
    metadata2 : pandas.DataFrame
        The metadata for second dataset.
    columns1 : str | list of str
        A subset of columns from the first dataset with metadata
        given by ``metadata1``.
    columns2 : str | list of str
        A subset of columns from the second dataset with metadata
        given by ``metadata2``.

    Returns
    -------
    bool
        ``True``.

    Raises
    ------
    AssertionError
        An error is thrown if, for each pair of corresponding
        join keys from ``columns1`` and ``columns2``,
        their sets of roles in the metadata are not identical.
    """

    # check
    assert_type(metadata1, pd.DataFrame)
    assert_type(metadata2, pd.DataFrame)
    assert_type_or_list_of_type(columns1, str)
    assert_type_or_list_of_type(columns2, str)

    # prepare
    columns1 = fmisc.maybe_make_list(columns1)
    columns2 = fmisc.maybe_make_list(columns2)

    # check
    assert_true(len(columns1) == len(columns2))
    assert_values_in_dataframe_column(metadata1, "variable", columns1)
    assert_values_in_dataframe_column(metadata2, "variable", columns2)

    # Verify that the set of roles for each
    # pair of corresponding variables is identical.
    for leftx, rightx in zip(columns1, columns2):
        # get the roles for the left and right join keys.
        roles_left = (
            metadata1.query("variable == @leftx").copy()["role"].tolist()
        )
        roles_right = (
            metadata2.query("variable == @rightx").copy()["role"].tolist()
        )

        # the roles must be identical.
        if set(roles_left) != set(roles_right):
            msg = (
                "A pair of corresponding columns have difference roles "
                "in the metadata:\n"
                + f"Left column: {leftx}\n"
                + f"Roles for the left column: {roles_left}\n"
                + f"Right column: {rightx}\n"
                + f"Roles for the right column: {roles_right}\n"
            )
            raise AssertionError(msg)

    return True


def assert_time_and_time_series_are_valid(data, time, time_series):
    """Verify that the columns representing time and time-series are valid.

    This helper function verifies that the columns
    that distinguish time-series within a dataset with
    multiple time-series and the column which describes
    time are valid.
    It checks that they are mutually disjoint and exist
    in the dataset.

    Parameters
    ----------
    data : pandas.DataFrame | None
        The dataset to which the ``time`` and ``time_series``
        pertain. It will be verified that the columns
        indicated in ``time`` and ``time_series`` exist
        in this dataframe. If ``None``, then no such
        check will be made.
    time : str | list of str
        The column which represents time.
        It will be verified that this column exists in
        ``data`` (if provided), and that it is not included
        in ``time_series`` (if provided).
    time_series : str | list of str | None
        The names of the columns to group by in order
        to distinguish distinct time series
        within the data. If ``None``, then all of the
        data in ``data`` should be from the same, single
        time-series.
        If not ``None``, then
        it will be verified that the columns exist
        in ``data`` and do not include ``time``.

    Returns
    -------
    bool
        ``True``.

    Raises
    ------
    AssertionError
        If any check fails.
    """

    # check
    assert_type(data, pd.DataFrame)
    assert_type_or_list_of_type(time, str)
    assert_type_or_list_of_type(time_series, str, allow_None=True)

    # time
    if isinstance(time, list):
        msg = (
            "Time must be indicated by exactly one column, "
            f"but {len(time)} were provided:\n{time}"
        )
        assert_true(len(time) == 1, msg=msg)

    assert_columns_in_dataframe(data, time)

    # time-series
    if time_series is not None:
        assert_columns_in_dataframe(data, time_series)

        if len(set(time_series) & set([time])) > 0:
            msg = (
                "The columns that distinguish time-series must "
                "be disjoint from the column that indicates time.\n"
                f"time series:\n{time_series}\n"
                f"time:\n{time}"
            )
            raise AssertionError(msg)

        if len(time_series) != len(set(time_series)):
            msg = (
                "Some of the columns in ``time_series`` "
                + f"are redundant.\ntime_series={time_series}"
            )
            raise AssertionError(msg)

    return True


def assert_lookback_offset_horizon_are_valid(lookback, offset, horizon):
    """Verifies that the lookback, offset, and horizon arguments are valid.

    Forecasting-like time-series problems can be formulated
    with three parameters: lookback, offset, and horizon.
    These three parameters must be integers representing
    the number of timesteps.
    The lookback must be >= 0, and the horizon >=1.
    This helper function verifies these properties.

    Parameters
    ----------
    lookback : float | int | None
        An integer representing the lookback of a step or model
        in terms of the number of timesteps.
        Must be >= 0.
    offset : float | int | None
        An integer representing the offset of a step or model
        in terms of the number of timesteps.
        It can be any integer value.
    horizon : float | int | None
        An integer representing the horizon of a step or model
        in terms of the number of timesteps.
        Must be >= 1.

    Returns
    -------
    bool
        ``True``.

    Raises
    ------
    AssertionError
        If any check fails.
    """

    # check type
    assert_type(lookback, [float, int], allow_None=True)
    assert_type(offset, [float, int], allow_None=True)
    assert_type(horizon, [float, int], allow_None=True)

    # must be integers
    if lookback is not None:
        lookback = fmisc.try_to_cast_to_integer(lookback)

    if offset is not None:
        offset = fmisc.try_to_cast_to_integer(offset)

    if horizon is not None:
        horizon = fmisc.try_to_cast_to_integer(horizon)

    # constraints
    if lookback is not None:
        assert_true(lookback >= 0)

    if horizon is not None:
        assert_true(horizon >= 1)

    return True


def assert_no_forbidden_arguments_provided(provided, forbidden):
    """Verify that no forbidden arguments were provided to kwargs.

    The user can provide engine-specific arguments to a model
    to tweak implementation-specific idiosyncratic arguments
    which are by definition not hyperparameters for the model.

    This helper function verifies that none of the engine-specific
    arguments provided via ``kwargs`` contain forbidden arguments.

    Parameters
    ----------
    provided : dict
        The engine-specific arguments provided as ``kwargs``
        to the model. Keys are strings giving the name of
        model arguments. These keys should not overlap with
        ``forbidden``.
    forbidden : list of str
        A list of arguments which the user is not allowed to
        specify via engine-specific arguments to ``kwargs``.

    Returns
    -------
    bool
        ``True``.

    Raises
    ------
    AssertionError
    """

    # check
    assert_type(provided, dict)
    assert_list_of_type(list(provided.keys()), str)
    assert_list_of_type(forbidden, str)

    overlapping_args = [z for z in forbidden if z in set(provided.keys())]

    if len(overlapping_args) > 0:
        msg = (
            "The engine-specific arguments specified values for "
            + "arguments which map to hyperparameters from the "
            + "model interface.\n"
            + "The offending keyword arguments are:\n"
            + f"{overlapping_args}."
        )
        raise AssertionError(msg)

    return True


def assert_uniform_sampling_rate(data, time, time_series):
    """Assert that the sampling rate is constant.

    This helper function will verify that the sampling rate
    is uniform in all time-series in the dataset.
    It does this by
        #. ordering records within each time-series by time;
        #. taking the lag-1 difference between successive timepoints
            within each time-series;
        #. verifying that the lag-1 difference is constant across
            all successive records across all time-series in the
            dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset whose sampling rate will be inferred.
    time : str
        The column in the ``data`` which represents time.
        Its dtype must be a timestamp.
    time_series : str | List[str] | None
        The columns which are used to distinguish time-series
        within the ``data``.
        For example, ``['patient_id', 'visit_id']``.
        Never includes the ``time`` column.
        Use ``None`` if all of records in the ``data``
        are from the same time-series.

    Returns
    -------
    float
        The sampling rate of the ``data``, in units of Hertz,
            inferred by sorting the data by ``time``
            and then taking the lag-1 difference of ``time`` between
            consecutive records within each ``time_series``.
    """

    # check
    assert_time_and_time_series_are_valid(data, time, time_series)

    if data.shape[0] < 2:
        msg = "Need more than one record to verify the sampling rate."
        raise AssertionError(msg)

    # prepare
    if isinstance(time, list):
        time = time[0]

    time_series = fmisc.maybe_make_list(time_series)

    # get the lag-1 difference between timestamps in ordered time-series.
    diff_np = fmisc.get_lag1_difference_of_timestamps(data, time, time_series)

    # count the number of each difference
    diffs, counts = np.unique(diff_np, return_counts=True)

    # all successive differences must be identical.
    if len(diffs) > 1:
        # get invalid indeces
        outlier_diff = diffs[np.argmin(counts)]
        (indeces_of_bad_diff,) = np.nonzero(diff_np == outlier_diff)
        indeces_to_print = np.concatenate(
            (indeces_of_bad_diff, (indeces_of_bad_diff + 1))
        )
        indeces_to_print = np.unique(indeces_to_print)

        # sort by time.
        sort_cols = [time] if time_series is None else time_series + [time]
        sub_df = data.sort_values(by=sort_cols).iloc[indeces_to_print, :]

        msg = (
            "Not all of the successive difference between "
            "timesteps are the same. Here are the counts of each "
            "successive difference:\n"
            f"unique values:\n{diffs.__str__()}\n"
            f"counts:\n{counts.__str__()}\n"
            "Data with invalid lag-1 differences in time:\n"
            f"{sub_df.__str__()}"
        )
        raise AssertionError(msg)

    # check that the diff is valid.
    if np.isnan(diffs[0]) or diffs[0] == 0:
        msg = (
            "Unable to infer a valid sampling rate.\n"
            "The inferred difference of successive timesteps "
            f"is {diffs[0]}."
        )
        raise AssertionError(msg)

    return True


def assert_instance(x, cls, allow_None=False, msg=None):
    """Verify that an object is an instance of a given class.
    This helper function verifies that an object is an instance
    of a specific class.
    The underlying workhorse is ``isinstance()``.
    Parameters
    ----------
    x : Any
        An object which will be verified to be an instance of
        a certain class or among a selection of classes given by ``cls``.
    cls : Any | list
        If a ``list`` is provided, then it should contain classes,
        in which case it will be verified that ``x`` is an instance of
        at least one of those classes.
        Otherwise, if will be verified directly that ``x`` is an instance
        of ``cls``.
    allow_None : bool
        If ``True``, then no error will be thrown if ``x is None``,
        regardless of what ``cls`` is.
        This is for convenience of passthrough.
    msg : str | None
        An message that will be prepended to any error message produced.
    Returns
    -------
    bool
        ``True``.
    Raises
    ------
    TypeError
        An error is thrown if ``x`` is not an instance of ``cls``
        or any of the classes listed therein.
    """

    # check
    assert_type(allow_None, bool)
    assert_type(msg, str, allow_None=True)

    # short-circuit
    if allow_None is True and x is None:
        return True

    # prepare
    if msg is None:
        msg = ""

    is_satisfied = True
    if isinstance(cls, list):
        is_satisfied = any([isinstance(x, z) for z in cls])
    else:
        is_satisfied = isinstance(x, cls)

    if not is_satisfied:
        msg = (
            msg + f"The object should be an instance of {cls}, "
            "but it is not.\n"
            f"The object's type is: {type(x)}\n"
            "The parents of the type of the object are: "
            f"{type(x).__bases__}\n"
        )
        raise TypeError(msg)

    return True


def assert_instance_or_list_thereof(x, cls, allow_None=False, msg=None):
    """Verify that an object is an instance of a given class or a list thereof.
    This helper function verifies that an object is an instance
    of a specific class, or the object is a list of objects, each of
    which is an instance of a given class.
    This function is basically a wrapper around ``assert_instance``
    to make the case of checking a list of objects (``x`` is a ``list``)
    more convenient.
    The underlying workhorses are ``isinstance()``
    and ``assert_instance()``.
    Parameters
    ----------
    x : Any
        An object or list of objects.
        If a ``list``, then each object in the list will be verified
        to be an instance of ``cls`` via ``assert_instance()``.
        Otherwise, ``x`` will be verified against ``cls`` directly
        via ``assert_instance()``.
    cls : Any | list
        If a ``list`` is provided, then it should contain classes,
        in which case it will be verified that ``x`` is an instance of
        at least one of those classes or a list of objects, each of which
        must be an instance of at least one of those classes.
        Otherwise, if will be verified directly that ``x`` is an instance
        of ``cls`` or a list of object, each of which is an
        instance of ``cls``.
    allow_None : bool
        If ``True``, then no error will be thrown if ``x is None``,
        regardless of what ``cls`` is.
        This is for convenience of passthrough.
    msg : str | None
        An message that will be prepended to any error message produced.
    Returns
    -------
    bool
        ``True``.
    Raises
    ------
    TypeError
        An error is thrown if ``x`` is not an instance of ``cls``
        or any of the classes listed therein.
    """

    # check
    assert_type(allow_None, bool)
    assert_type(msg, str, allow_None=True)

    # short-circuit
    if allow_None is True and x is None:
        return True

    # prepare
    if msg is None:
        msg = ""

    if isinstance(x, list):
        for z in x:
            assert_instance(x=z, cls=cls, allow_None=allow_None, msg=msg)
    else:
        assert_instance(x=x, cls=cls, allow_None=allow_None, msg=msg)

    return True


def assert_subclass(x, cls, allow_None=False, msg=None):
    """Verify that an object is a subclass of a given class.
    This helper function verifies that an object is a subclass
    of a specific class.
    The underlying workhorse is ``issubclass()``.
    Parameters
    ----------
    x : Any
        An object which will be verified to be a subclass of
        a certain class or among a selection of classes given by ``cls``.
    cls : Any | list
        If a ``list`` is provided, then it should contain classes,
        in which case it will be verified that ``x`` is a subclass of
        at least one of those classes.
        Otherwise, if will be verified directly that ``x`` is a subclass
        of ``cls``.
    allow_None : bool
        If ``True``, then no error will be thrown if ``x is None``,
        regardless of what ``cls`` is.
        This is for convenience of passthrough.
    msg : str | None
        An message that will be prepended to any error message produced.
    Returns
    -------
    bool
        ``True``.
    Raises
    ------
    TypeError
        An error is thrown if ``x`` is not a subclass of ``cls``
        or any of the classes listed therein.
    """

    # check
    assert_type(allow_None, bool)
    assert_type(msg, str, allow_None=True)

    # short-circuit
    if allow_None is True and x is None:
        return True

    # prepare
    if msg is None:
        msg = ""

    is_satisfied = True
    if isinstance(cls, list):
        is_satisfied = any([issubclass(x, z) for z in cls])
    else:
        is_satisfied = issubclass(x, cls)

    if not is_satisfied:
        msg = (
            msg + f"The object should be a subclass of {cls}, "
            "but it is not.\n"
            f"The object's type is: {type(x)}\n"
            "The parents of the type of the object are: "
            f"{type(x).__bases__}\n"
        )
        raise TypeError(msg)

    return True


def assert_subclass_or_list_thereof(x, cls, allow_None=False, msg=None):
    """Verify that an object is a subclass of a given class or a list thereof.
    This helper function verifies that an object is a subclass
    of a specific class, or the object is a list of objects, each of
    which is a subclass of a given class.
    This function is basically a wrapper around ``assert_subclass``
    to make the case of checking a list of objects (``x`` is a ``list``)
    more convenient.
    The underlying workhorses are ``issubclass()``
    and ``assert_subclass()``.
    Parameters
    ----------
    x : Any
        An object or list of objects.
        If a ``list``, then each object in the list will be verified
        to be a subclass of ``cls`` via ``assert_subclass()``.
        Otherwise, ``x`` will be verified against ``cls`` directly
        via ``assert_subclass()``.
    cls : Any | list
        If a ``list`` is provided, then it should contain classes,
        in which case it will be verified that ``x`` is a subclass of
        at least one of those classes or a list of objects, each of which
        must be a subclass of at least one of those classes.
        Otherwise, if will be verified directly that ``x`` is a subclass
        of ``cls`` or a list of object, each of which is an
        subclass of ``cls``.
    allow_None : bool
        If ``True``, then no error will be thrown if ``x is None``,
        regardless of what ``cls`` is.
        This is for convenience of passthrough.
    msg : str | None
        An message that will be prepended to any error message produced.
    Returns
    -------
    bool
        ``True``.
    Raises
    ------
    TypeError
        An error is thrown if ``x`` is not a subclass of ``cls``
        or any of the classes listed therein.
    """

    # check
    assert_type(allow_None, bool)
    assert_type(msg, str, allow_None=True)

    # short-circuit
    if allow_None is True and x is None:
        return True

    # prepare
    if msg is None:
        msg = ""

    if isinstance(x, list):
        for z in x:
            assert_subclass(x=z, cls=cls, allow_None=allow_None, msg=msg)
    else:
        assert_subclass(x=x, cls=cls, allow_None=allow_None, msg=msg)

    return True
