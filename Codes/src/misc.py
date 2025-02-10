from itertools import chain

import numpy as np
import pandas as pd

import assertions as fast


def get_column_names_with_role(df, metadata, roles):
    """Get the names of the columns that have a given role."

    This is a convenience method for getting the names
    of the columns in a DataFrame that have at least
    one of the indicated roles.
    Note: a column need not have all of the roles
    specified in ``roles``.

    Parameters
    ----------
    df : pd.DataFrame
        Any DataFrame.
    metadata : pd.DataFrame
        Metadata about the columns of ``df``; specifically,
        it must have a column called ``role``.
    roles : str or list of str
        The roles of interest. A column name of ``df``
        will included in the returned column names
        if it has any of the roles in ``roles``; i.e.
        it is not required to have all of the roles.

    Returns
    -------
    list of str, or None
        A list of the column names of ``df`` which have
        any of the roles in ``roles`` is returned.
        ``None`` is returned if no columns in ``df`` have
        any of the roles in ``roles``.
    """

    # check
    fast.assert_type(df, pd.DataFrame)
    fast.assert_type_or_list_of_type(roles, str)
    fast.assert_metadata_is_valid(metadata, df)

    cn = metadata.query("role == @roles")["variable"].unique().tolist()
    fast.assert_true(
        len(cn) > 0,
        msg=f"Utils Error: No columns with any of these roles: {roles}",
    )
    fast.assert_columns_in_dataframe(df, cn)

    return cn


def maybe_make_list(x):
    """Make a list if the object is not a list already.

    Very similar to the function of the same named in pandas,
    this helper function will put an object in a list
    if it is not a list already. If the object is ``None``,
    it will be returned as-is.

    Parameters
    ----------
    x : Anything
        Any object. Can be a list or not, or ``None``.

    Returns
    -------
    list or None
    """

    if x is None or isinstance(x, list):
        return x
    else:
        return [x]


def make_safe_column_name(df, name=None):
    """Generate a safe name for a new column.

    This function generates a name for a new column
    in the dataframe which does not clash with
    any of the existing column names in the dataframe.
    This is useful when you want to add columns to a
    dataframe but have to be sure that they don't overwrite
    an existing column names, e.g. provided by a dataset or user.

    Parameters
    ----------
    df : pandas.DataFrame | list of pd.DataFrame
        Any dataframe or list of dataframes.
    name : str | None
        A desired name for the new column name.
        If there is already a column named ``name``, then
        a new name will be found which is somewhat similar
        to ``name``.

    Returns
    -------
    str
        A name for a new column which is not among
        any of the existing column names in the data frame.
    """

    fast.assert_type_or_list_of_type(df, pd.DataFrame)
    fast.assert_type(name, str, allow_None=True)

    dfs = maybe_make_list(df)
    cn = list(
        set(list(chain.from_iterable([z.columns.tolist() for z in dfs])))
    )

    proposal = name
    ctr = 0
    while proposal in cn:
        ctr = ctr + 1
        proposal = f"{name}_{ctr}"

    if proposal in cn:
        msg = "Unable to generate a distinct column name."
        msg = msg + "How did this happen?"
        raise AssertionError(msg)

    return proposal


def get_subset_by_role(data, metadata, roles):
    """Get subset of dataframe by role.

    This helper function returns the subset of the
    DataFrame whose columns have certain roles.

    Parameters
    ----------
    data : pandas.DataFrame
        Any dataframe.
    metadata : pandas.DataFrame
        The metadata for ``data``.
    roles : str | list of str
        The roles of interest.
        If a column has at least one of the roles listed
        herein, it will be included in the result.

    Returns
    -------
    pandas.DataFrame
        The subset of ``data`` with only the
        columns whose role are in ``roles``.
    """

    # check
    fast.assert_type(data, pd.DataFrame)
    fast.assert_type(metadata, pd.DataFrame)
    fast.assert_columns_in_dataframe(metadata, ["role", "variable"])
    fast.assert_type_or_list_of_type(roles, str)
    rl = maybe_make_list(roles)
    fast.assert_values_in_dataframe_column(metadata, "role", rl)

    mc = metadata.query(f"role.isin({rl})").get("variable").unique().tolist()

    fast.assert_type_or_list_of_type(mc, str)
    fast.assert_columns_in_dataframe(data, mc)

    mdf = data[mc]

    return mdf


def are_dataframes_alignable(df1, metadata1, df2, metadata2):
    """Determine if two dataframe can be aligned by their identifier columns.

    This method will determine if two dataframes
    could be aligned by their identifiers.
    The criteria are:

    - must have the same exact set of identifier columns
    (though column order does not matter).
    - must have the same number of rows.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Any dataframe with columns that have role ``'identifier'``.
    df2 : pandas.DataFrame
        Any dataframe with columns that have role ``'identifier'``.
    metadata1 : pandas.DataFrame
        metadata for ``df1``.
    metadata2 : pandas.DataFrame
        metadata for ``df2``.

    Returns
    -------
    bool
        ``True`` means that these two dataframes could conceivably
        be aligned by sorting the rows by the values of the identifier
        columns.
    """

    # check
    fast.assert_type(df1, pd.DataFrame)
    fast.assert_type(df2, pd.DataFrame)
    fast.assert_type(metadata1, pd.DataFrame)
    fast.assert_type(metadata2, pd.DataFrame)
    fast.assert_metadata_is_valid(metadata1, df1)
    fast.assert_metadata_is_valid(metadata2, df2)

    if df1.shape[0] != df2.shape[0]:
        msg = "The number of rows are unequal:\n"
        msg = msg + f"df1: {df1.shape[0]}\n"
        msg = msg + f"df2: {df2.shape[0]}\n"
        msg = msg + f"df1 = \n{df1.__str__()}\n"
        msg = msg + f"df2 = \n{df2.__str__()}\n"
        raise AssertionError(msg)

    # get identifiers
    id1 = sorted(
        metadata1.query('role == "identifier"')["variable"].unique().tolist()
    )
    id2 = sorted(
        metadata2.query('role == "identifier"')["variable"].unique().tolist()
    )

    fast.assert_type_or_list_of_type(id1, str)
    fast.assert_type_or_list_of_type(id2, str)

    # verify identifiers
    if id1 != id2:
        msg = "The identifier columns in the two dataframes do not match.\n"
        msg = (
            msg
            + f"The identifier columns in the first dataframe are: {id1}.\n"
        )
        msg = (
            msg
            + f"The identifier columns in the second dataframe are: {id2}.\n"
        )
        raise AssertionError(msg)

    fast.assert_columns_in_dataframe(df1, id1)
    fast.assert_columns_in_dataframe(df2, id2)

    return True


def align_dataframes_by_identifiers(df1, metadata1, df2, metadata2):
    """Align two dataframes by their identifier columns.

    This method will sort two dataframes so that
    for corresponding rows between the two dataframes,
    the values in the identifier columns are identical.
    The dataframes must have the same number of rows
    and same identifier columns.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Any dataframe with columns that have role ``'identifier'``.
    df2 : pandas.DataFrame
        Any dataframe with columns that have role ``'identifier'``.
    metadata1 : pandas.DataFrame
        metadata for ``df1``.
    metadata2 : pandas.DataFrame
        metadata for ``df2``.

    Returns
    -------
    tuple
        A tuple of length 2 of dataframes.
        The first entry is a sorted version of ``df1``.
        The second entry is a sorted version of ``df2``.
        These two dataframes have identical values
        in the identifier columns in each row.
    """

    # check
    res = are_dataframes_alignable(df1, metadata1, df2, metadata2)
    fast.assert_type(res, bool)

    if res is False:
        msg = "These two dataframes cannot be aligned."
        raise AssertionError(msg)

    id1 = sorted(
        metadata1.query("role == 'identifier'")
        .get("variable")
        .unique()
        .tolist()
    )

    id2 = sorted(
        metadata2.query("role == 'identifier'")
        .get("variable")
        .unique()
        .tolist()
    )

    # must do ``.reset_index()`` becuase ``.equals()`` considers the index.
    df1 = df1.sort_values(by=id1).reset_index(drop=True)
    df2 = df2.sort_values(by=id2).reset_index(drop=True)

    if df1[id1].equals(df2[id2]) is not True:
        # get disagreeing rows.
        mdf = df1[id1].merge(
            right=df2[id2],
            left_on=id1,
            right_on=id2,
            how="outer",
            indicator=True,
        )
        ldf = mdf[mdf["_merge"] == "left_only"]
        rdf = mdf[mdf["_merge"] == "right_only"]

        # error messsage
        msg = "The values in the identifier columns for the "
        msg = msg + "two dataframes are not identical after "
        msg = msg + "sorting.\n"
        msg = msg + f"id1 = {id1}\nid2 = {id2}\n"
        msg = msg + f"df1[id1]:\n{df1[id1].__str__()}\n"
        msg = msg + f"df2[id2]:\n{df2[id2].__str__()}\n"
        msg = msg + f"Left only:\n{ldf.__str__()}\n"
        msg = msg + f"Right only:\n{rdf.__str__()}\n"
        raise AssertionError(msg)

    return (df1, df2)


def generate_contract(df):
    """Generates input and output contracts based on the passed dataframe
        The dataframe for which to create the contract metadata

    Returns
    -------
    pandas.DataFrame
        The dataframe contract
    """
    contract = pd.DataFrame(
        {
            "variables": df.columns.tolist(),
            "dtype": [z.name for z in df.dtypes.tolist()],
            "rank": list(range(df.shape[1])),
        }
    )
    return contract


def try_to_cast_to_integer(x, allow_None=False, msg=None):
    """Cast a float to an integer, if possible.

    This method tries to cast an integer-like value
    to an integer. If a float value is provided
    and it is integer-like, then
    it will be cast to integer and return it.
    Integer inputs will be returned as-is.
    Anything else will throw an error.

    Parameters
    ----------
    x : float | int | list of float or int | tuple of float or int | None
        A numeric value which is hopefully integer-like;
        or a tuple or list of such numeric values.
        Such numeric values will be cast to integer, if possible,
        and returned as scalar, list, or tuple, as per the input
        data structure.
    allow_None : bool
        If ``True`` and ``x`` is ``None``, then it
        will be returned as-is.
        Otherwise, if ``x`` is ``None``, then an error
        will be thrown.
        Only applies if ``x`` is ``float`` or ``int``, i.e.
        does not propagate into converting a list or tuple
        of numeric values.

    Returns
    -------
    int | list | tuple | None
        The casting of ``x`` to ``int``, if it can be.
        If a list or tuple was provided, then a list or tuple
        of ``int`` will be returned, respectively.

    Raises
    ------
    TypeError
    """

    # check
    fast.assert_type(allow_None, bool)

    # check for None
    if allow_None is True:
        if x is None:
            return x
    else:
        if x is None:
            msg = "An integer-like value is required, but received ``None``."
            raise TypeError(msg)

    # try to cast as ``int``.
    if isinstance(x, int):
        return x
    elif isinstance(x, float):
        if x.is_integer() is True:
            return int(x)
    elif isinstance(x, list):
        x = [try_to_cast_to_integer(z) for z in x]
        return x
    elif isinstance(x, tuple):
        x = tuple((try_to_cast_to_integer(z) for z in x))
        return x

    if msg is None:
        msg = "An integer-like value must be provided, "
        msg = msg + f"but instead the following was provided:\n{x}\n"

    raise TypeError(msg)


def container_obj(container):
    """Returns container's object depending on type of container

    Parameters
    ----------
    container : object
        container to retrieve object from
    """
    container_type = type(container).__name__

    if container_type == "StepContainer":
        return container._step
    elif container_type == "ModelContainer":
        return container._model
    elif container_type == "DecisionFunctionContainer":
        return container._decision_function
    else:
        raise TypeError()


def infer_sampling_rate(data, time_series, time, strict):
    """Infer the sampling rate of a dataset.

    This method will take a time-series dataset,
    sort by the column representing time,
    and then infer the sampling rate of the data by taking the
    mode of the differences between consecutive timesteps
    within each time-series.

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
    strict : bool
        if ``True``, then an error will be thrown unless the inferred
        sampling rate is exactly the same across all records in
        the ``data``;
        otherwise, the mode of the lag-1 difference of ``time``
        between consecutive records within each ``time_series``
        will be returned.

    Returns
    -------
    float
        The sampling rate of the ``data``, in units of Hertz,
            inferred by sorting the data by ``time``
            and then taking the lag-1 difference of ``time`` between
            consecutive records within each ``time_series``.
    """

    # check
    fast.assert_type(data, pd.DataFrame)
    fast.assert_type(time, str)
    fast.assert_type_or_list_of_type(time_series, str, allow_None=True)
    fast.assert_type(strict, bool)

    fast.assert_true(
        data.shape[0] > 1,
        msg=(
            "Need more than one record to infer sampling rate "
            "from consecutive timesteps"
        ),
    )

    # prepare
    time_series = maybe_make_list(time_series)

    # get lag-1 differences in ``time``.
    if time_series is not None:
        # sort to prepare for lag-1 difference operation.
        sub_df = data[time_series + [time]].sort_values(
            by=time_series + [time]
        )

        # lag-1 difference per time-series.
        diff_col = make_safe_column_name(data, "_diff")
        sub_df[diff_col] = sub_df.groupby(by=time_series).diff(periods=1)

        # drop the first element (undefined lag-1 difference)
        # for each time-series.
        diff_np = (
            sub_df.groupby(by=time_series)
            .apply(lambda df: df.iloc[1:])
            .loc[:, diff_col]
            .to_numpy()
        )

    else:
        diff_np = np.diff(np.sort(data[time].to_numpy()))

    # sanity check
    fast.assert_true(diff_np.shape[0] > 0)

    # count the number of each difference
    diffs, counts = np.unique(diff_np, return_counts=True)

    if strict is True:
        # all successive differences must be identical.
        if len(diffs) > 1:
            msg = (
                "Not all of the successive difference between "
                "timesteps are the same. Here are the counts of each "
                "successive difference:\n"
                f"unique values:\n{diffs.__str__()}\n"
                f"counts:\n{counts.__str__()}"
            )
            raise AssertionError(msg)

        mode_diff = diffs[0]
    else:
        # take the mode of differences.
        descend_index = np.argsort(-counts)
        mode_diff = diffs[descend_index[0]]

    # check that the diff is valid.
    try:
        if np.isnan(mode_diff) or mode_diff == 0:
            msg = (
                "Unable to infer a valid sampling rate.\n"
                "The inferred difference of successive timesteps "
                f"is {mode_diff}."
            )
            raise AssertionError(msg)
    except Exception as e:
        msg = e + " --- Perhaps your time column is still typed as an object?"
        raise AssertionError(msg)
    # infer sampling rate from sampling period.
    sampling_rate = pd.Timedelta(value=1, unit="seconds") / mode_diff

    return sampling_rate


def overlappingwindows_timebased(df, sampling_rate, time, stride, lookback):
    """
    Using a date/time type, identify segments of data as overlapping lookbacks.
    For example, if we need to output a HRV value every 1 minute, but it takes
    5 minutes to produce an accurate value, then we would need a 5 minute
    lookback with a 4 minute overlap.

    This function can handle irregularly spaced data
    as it is operating strictly on the timestamps and not referencing sample
    count within each lookback.

    Parameters
    ----------
    df: dataframe
        Input dataframe to be chunked out into segments
    sampling_rate: float
        Sampling rate of data provided within df; used for data
        availability calculations
    time: str
        Name of the time column to be used for the basis of all segmentation
    stride: int
        Number of seconds to increment each lookback of data by (i.e. for 1
        minute stride, provide stride=60)
        For the purposes of producing a value every SAMPLE, this should be
        the inverse of the sampling rate
    lookback: int
        Number of seconds for each lookback of data to be processed (i.e. for
        a 5 minute lookback of data, provide lookback=60*5)

    Returns
    ----------
    overlapping_df_list: list of datarames, list of length M
        List of each dataframe segment to be processed individually
        NOTE: the timestamps used on the comparison are always start
        INCLUSIVE and end EXCLUSIVE
    timestamping_df_list: list
        list of the timestamps associated with each overlapping_df_list
        segment, list of length M
    da_df_list: list
        List of data avilability ratio for each lookback of data
        (i.e. 0-1, where 1 is 100% of data present; 0 is 0% of data present)
    """
    # Resample to 1 stride
    time_basis = pd.DataFrame(
        {
            "end_t": pd.date_range(
                df[time].min().floor("s") - pd.Timedelta(f"{lookback}s"),
                df[time].max().ceil("s") + pd.Timedelta(f"{lookback}s"),
                freq=f"{int(1e9*(stride))}N",
            )
        }
    )
    # Add the ending time using the lookback size
    time_basis["start_t"] = time_basis["end_t"] - pd.Timedelta(f"{lookback}s")

    overlapping_df_list = []
    timestamping_df_list = []
    da_df_list = []
    for ind, ind_df in time_basis.iterrows():
        df_of_lookback = df[
            df[time].between(
                ind_df["start_t"], ind_df["end_t"], inclusive="left"
            )
        ]
        overlapping_df_list.append(df_of_lookback)
        timestamping_df_list.append(ind_df["end_t"])
        da_df_list.append((len(df_of_lookback) / lookback) / sampling_rate)

    return overlapping_df_list, timestamping_df_list, da_df_list


def format_required_packages(requirements):
    """Format a list of required packages into a dataframe.

    For simplicity and convenience, Steps, Models, and Decision functions
    provide required packages as a list of 2-tuples of format:
    <(package, language)>
    where each entry is a string.

    This convenience function reformats the list of 2-tuples
    into a dataframe.

    Parameters
    ----------
    requirements : list | None
        A list of 2-tuples where the first entry is a string giving
        the name of a required package, and the second entry is a string
        giving the name of the language in which that package is implemented.
        If ``None``, then will return a dataframe with no rows.

    Returns
    -------
    pandas.DataFrame
        A dataframe of the required packages.
        The columns are: package, language.
        Each entry in each column is a string.
        The rows are unique.
        If ``requirements is None``, then the dataframe will have
        no rows.
    """

    # check
    fast.assert_list_of_type(requirements, tuple, allow_None=True)

    if requirements is None:
        return pd.DataFrame(columns=["package", "language"])

    # make two agglomerated lists of same length.
    packages = []
    languages = []

    for entry in requirements:
        # check
        fast.assert_true(len(entry) == 2)
        fast.assert_type(entry[0], str)
        fast.assert_type(entry[1], str)

        # append
        packages.append(entry[0])
        languages.append(entry[1])

    # make dataframe from the lists.
    reqs = pd.DataFrame({"package": packages, "language": languages})

    reqs.drop_duplicates(inplace=True)

    return reqs


def resolve_lookback_to_number_of_timesteps(lookback, sampling_rate):
    """Get the representation of lookback as number of timesteps.

    Sometimes the lookback may be provided as an absolute amount of time,
    such as a string representing a ``pandas`` timedelta or
    ``pandas`` time offset, yet it is necessary to represent
    the lookback as the number of previous timesteps, assuming a
    regular sampling rate.

    Given a non-integer representation of a lookback, this function will
    return an integer representing the number of previous timesteps.
    The integer representation does not include the current timestep,
    so that a lookback of 1 means that the previous timestep if used
    in addition to the current timestep.

    This operation is the inverse of ``resolve_lookback_to_window_of_time``.

    Parameters
    ----------
    lookback : float | int | str | pandas.Timedelta | None
        - ``None`` will be returned as-is.
        - ``float`` or ``int`` are assumed to already represent
        the number of previous timesteps, and will be verified
        to be a positive integer.
        - ``str`` must represent a timedelta or
        time offset from ``pandas``
    sampling_rate : float | int
        The sampling rate of the date in units of Hertz.
        The timesteps of the data are assumed to be uniform.

    Returns
    -------
    int | None
        ``None`` if and only if ``looback=None``;
        otherwise:
        a positive integer giving the lookback as the number
        of previous timesteps.
        The integer representation does not include the current timestep,
        so that a lookback of 1 means that the previous timestep if used
        in addition to the current timestep.
    """

    # check
    fast.assert_type(
        lookback, [float, int, str, pd.Timedelta], allow_None=True
    )
    fast.assert_type(sampling_rate, [float, int])
    fast.assert_true(sampling_rate > 0)

    # resolve
    if lookback is None:
        # passthrough
        return lookback
    elif isinstance(lookback, float) or isinstance(lookback, int):
        # verify it is a positive integer
        lookback = try_to_cast_to_integer(lookback)
        fast.assert_true(lookback > 0)
        return lookback
    else:
        # resolve from time to positive integer
        fast.assert_non_negative_timedelta(lookback)

        if isinstance(lookback, str):
            lookback = pd.Timedelta(lookback)

        frequency = 1.0 / sampling_rate
        lookback_seconds = lookback.total_seconds()
        num_steps = lookback_seconds / frequency

        msg = (
            f"The lookback value {lookback} is not divisible "
            + f"by the sampling rate {sampling_rate}."
        )
        num_steps = try_to_cast_to_integer(num_steps, allow_None=True, msg=msg)
        fast.assert_true(num_steps > 0)

        return num_steps


def resolve_lookback_to_window_of_time(lookback, sampling_rate):
    """Get the representation of lookback as the width of a window into the
    past.

    This function computes the length of the lookback window
    from the lookback represented as the number of previous timesteps.
    A regular sampling rate is assumed.

    The integer representation as the number of previous timesteps
    does not include the current timestep,
    so that a lookback of 1 means that the previous timestep if used
    in addition to the current timestep.

    This operation is the inverse of
    ``resolve_lookback_to_number_of_timesteps``.

    Parameters
    ----------
    lookback : float | int | str | pandas.Timedelta | None
        - ``None`` will be returned as-is.
        - ``float`` or ``int`` represent the number of previous timesteps
        and will be used to infer the length of the lookback window
        as a stretch of time via the ``sampling_rate``.
        - ``str`` or ``pandas.Timedelta`` are assumed to already
        represent the lookback window, and will be returned as-is.
    sampling_rate : float | int
        The sampling rate of the date in units of Hertz.
        The timesteps of the data are assumed to be uniform.

    Returns
    -------
    pandas.Timedelta | None
        ``None`` if and only if ``looback=None``;
        otherwise:
        the length of the lookback window
        in units of seconds.
    """

    # check
    fast.assert_type(
        lookback, [float, int, str, pd.Timedelta], allow_None=True
    )
    fast.assert_type(sampling_rate, [float, int])
    fast.assert_true(sampling_rate > 0)

    # resolve
    if lookback is None:
        return lookback
    elif isinstance(lookback, str):
        lookback = pd.Timedelta(lookback)
    elif not isinstance(lookback, pd.Timedelta):
        # int, float.
        lookback = try_to_cast_to_integer(lookback)
        fast.assert_true(lookback > 0)

        # convert to window of time
        frequency = 1.0 / sampling_rate
        lookback = frequency * lookback
        lookback = pd.Timedelta(value=lookback, unit="seconds")

    # sanity check
    fast.assert_type(lookback, pd.Timedelta)
    fast.assert_non_negative_timedelta(lookback)

    return lookback


def make_hyperparameter_uuids(dataframe):
    """Make UUIDs for hyperparameters.

    Many higher-level objects, such as ``Recipe`` and ``ModelWorkflow``,
    have a ``.get_hyperparameters()`` method which
    returns a dataframe which one row per hyperparameter, where
    the hyperparameters are from various *flamingo* objects;
    e.g. Steps, Models, Decision Functions.
    This method creates a UUID for each hyperparameter
    (row in the dataframe) by using the values of the other
    properties of the object to which the hyperparameter pertains,``
    and which are captured in the dataframe returned by
    ``.get_hyperparameters()``. These other properties are:
    component, rank.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        A dataframe with hyperparameter information as returned
        by the ``.get_hyperparameters()`` method which is common
        across many object (Recipe, ModelWorkflow, ...)

    Returns
    -------
    pandas.Series
        A Series of strings whose length is the same as the number
        of rows of ``dataframe`` and whose entries are in
        correspondence with the rows in ``dataframe``.
        Each entry in the list is a string giving a UUID
        for that hyperparameter.
        The UUID is unique with respect to all hyperparameters
        from all objects at the ``Recipe`` and ``ModelWorkflow`` levels.
    """

    # check
    fast.assert_type(dataframe, pd.DataFrame)
    fast.assert_columns_in_dataframe(
        dataframe, ["component", "rank", "hyperparameter"]
    )

    uuids = (
        dataframe["component"]
        + "_"
        + dataframe["rank"].astype(str)
        + "_"
        + dataframe["hyperparameter"]
    )

    # check
    fast.assert_type(uuids, pd.Series)
    fast.assert_true(len(uuids) == len(set(uuids)))

    return uuids


def get_lag1_difference_of_timestamps(data, time, time_series):
    """Get the lag-1 difference between timestamps within each time-series.

    This helper function will compute the lag-1 difference between
    consecutive timestamps within each time-series in the data.
    It does this by:
        #. ordering records within each time-series by time;
        #. taking the lag-1 difference between successive timepoints
            within each time-series;

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
    nump.array
        A vector giving the lag-1 difference between timestamps
        of consecutive records within each time-series, after sorting
        the input ``data`` by ``time_series`` (if not ``None``)
        and ``time``.
        The length of this vector is strictly less than the
        number of rows in ``data`` because the lag-1 difference
        is undefined for the first element in a time-series,
        and is therefore excluded from the output rather than filled
        with ``NaN``.
    """

    # check
    fast.assert_time_and_time_series_are_valid(data, time, time_series)

    # prepare
    if isinstance(time, list):
        time = time[0]

    time_series = maybe_make_list(time_series)

    # get lag-1 differences in ``time``.
    if time_series is not None:
        # sort to prepare for lag-1 difference operation.
        sub_df = (
            data[time_series + [time]]
            .sort_values(by=time_series + [time])
            .copy()
        )

        # lag-1 difference per time-series.
        diff_col = make_safe_column_name(data, "_diff")
        sub_df[diff_col] = sub_df.groupby(by=time_series).diff(periods=1)

        # drop the first element for each time-series because
        # the lag-1 difference is undefined for the first element
        # in the ordered time-series.
        diff_np = (
            sub_df.groupby(by=time_series)
            .apply(lambda df: df.iloc[1:])
            .loc[:, diff_col]
            .to_numpy()
        )

    else:
        diff_np = np.diff(np.sort(data[time].to_numpy()))

    # sanity check
    fast.assert_true(len(diff_np.shape) == 1)
    fast.assert_true(diff_np.shape[0] > 0)

    return diff_np
