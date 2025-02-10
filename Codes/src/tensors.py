from operator import xor

import numpy as np
import pandas as pd

from assertions import (
    assert_type_or_list_of_type,
    assert_type,
    assert_true,
    assert_columns_in_dataframe,
    assert_columns_are_numeric,
    assert_time_and_time_series_are_valid,
    assert_lookback_offset_horizon_are_valid,
    assert_metadata_is_valid,
    assert_uniform_sampling_rate,
)
from misc import (
    try_to_cast_to_integer,
    make_safe_column_name,
    maybe_make_list,
    get_column_names_with_role,
)


def make_lookback_tensor_for_multiple_time_series(
    data,
    time_series,
    time,
    features,
    lookback,
    offset,
    horizon,
    complete,
):
    """Converts a dataframe with multiple time-series into a 3d tensor.
    Deep learning models require a tensor as input.
    Deep learning models for time-series data with multiple features
    in general require a 3d tensors where one dimension represents the
    lookback; i.e. values of the features at previous timepoints.
    This method converts time-series data from a dataframe representation
    to a 3d tensor that is compatible with input to a deep learning model.
    The time-series data may contain multiple time-series with multiple
    features. The shape of the output 3d tensor is: (samples, features,
     lookback+1),
    where:
    - samples = the number of timepoints in the data, after dropping
        early timepoints which do not have a complete lookback
        history according to ``complete``,
        and later timepoints which do not have a full *prediction window*
        as specified by ``offset`` and ``horizon``.
    - features = the number of features in the data.
    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe of time-series data to be converted into
        tensor format. Lagged feature values should not have been
        made into columns (features) yet;
        i.e. don't use ``LagAsFeatures`` before this.
    time_series : str | list of str
        Columns of the time-series dataframe ``data`` which
        can be used to distinguish distinct time-series within
        the data. For example, if ``data`` contains time-series
        data from multiple patients, then perhaps
        ``time_series=['patient_id', 'visit_id']``.
    time : str
        The column in the time-series dataframe ``data`` which
        represents time. ``data`` will be sorted by the ``time``
        column in order to create the lagged features for
        the tensor.
    features : str | list of str
        The columns of the dataframe ``data`` which contain
        features. All other columns will be ignored, except for the
        columns in ``time_series`` and ``time`` which are used to arrange
        the tensor and identify appropriate lagged timepoints.
    lookback : float | int
        An integer giving the number of previous timepoints to consider
        at each timepoint *t*.
        ``lookback=0`` means that only the values of the features from
        the current timepoint *t* are considered at timepoint *t*;
        `lookback=1`` means that at timepoint ``t``, the values of the features
        at timepoints ``t`` and ``t-1`` are considered; and so on.
    offset : float | int | None
        An integer. Can be positive or negative.
        Given a timepoint *t* (an integer, not timestamp),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins ``offset`` timesteps
        after the current timepoint *t*.
        For training, ``offset`` and ``horizon`` should
        be provided so that the tensor of the features (``X``)
        and the tensor of the targets (``y``) have corresponding
        entries.
        For prediction on new data, ``offset`` and ``horizon``
        should probably be set to ``None``.
    horizon : float | int | None
        An integer.
        Given a timepoint *t* (an integer),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins ``offset`` timesteps
        after the current timepoint *t*.
        For training, ``offset`` and ``horizon`` should
        be provided so that the tensor of the features (``X``)
        and the tensor of the targets (``y``) have corresponding
        entries.
        For prediction on new data, ``offset`` and ``horizon``
        should probably be set to ``None``.
    complete : bool
        if ``True``, then the output tensor only contain samples
        for which all of the previous ``lookback`` samples exist
        in the input dataframe ``data``.
        If ``False``, then the size of the first dimension of the tensor
        will be exactly the number of rows in the dataframe ``data``,
        and correspond accordingly.
    Returns
    -------
    numpy.ndarray
        A 3-dimensional numpy array; i.e. a tensor.
        The shape of the output tensor is:
            ``(samples, features, lookback + 1)``,
        where:
        - ``samples`` =
            some number not greater than the number of rows
            in the data ``data``; it can be less
            if ``complete=True`` or if ``offset`` and ``horizon``
            are not ``None`.
        - ``features`` = the number of features in the data.
        - ``lookback + 1`` because ``lookback=0`` corresponds to
            only including the values of the features at the
            current timepoint *t* when creating a tensor
            for timepoint *t*.
            - ``tensor[:,:,0]`` corresponds to timepoint *t - lookback*;
            - ``tensor[:,:,1]`` corresponds to timepoint *t - lookback + 1*;
            - ...
            - ``tensor[:,:,lookback+1]`` corresponds to timepoint *t*.
    """

    # check
    assert_type(data, pd.DataFrame)

    assert_type_or_list_of_type(time_series, str)
    assert_columns_in_dataframe(data, time_series)
    time_series = maybe_make_list(time_series)

    assert_type(time, str)
    assert_columns_in_dataframe(data, time)
    if time in time_series:
        msg = "The ``time`` column should not be among the "
        msg = msg + "``time_series`` columns.\n"
        msg = msg + f"time: {time}\n"
        msg = msg + f"time_series: {time_series}"
        raise AssertionError(msg)

    assert_type_or_list_of_type(features, str)
    features = maybe_make_list(features)
    assert_true(len(features) > 0)
    assert_columns_in_dataframe(data, features)
    assert_columns_are_numeric(data, features)

    if time in features:
        msg = "The ``time`` column should not be among the "
        msg = msg + "``features`` columns.\n"
        msg = msg + f"time: {time}\n"
        msg = msg + f"features: {features}\n"
        raise AssertionError(msg)

    if any([z in features for z in time_series]):
        msg = "The ``time_series`` and `features`` columns should "
        msg = msg + "be disjoint.\n"
        msg = msg + f"time_series: {time_series}\n"
        msg = msg + f"features: {features}\n"
        raise AssertionError(msg)

    assert_type(lookback, [float, int])
    lookback = try_to_cast_to_integer(lookback)
    assert_true(lookback >= 0)

    assert_type(offset, [float, int], allow_None=True)
    if offset is not None:
        offset = try_to_cast_to_integer(offset)
        # offset is allowed to be negative.

    assert_type(horizon, [float, int], allow_None=True)
    if horizon is not None:
        horizon = try_to_cast_to_integer(horizon)
        assert_true(horizon >= 0)

    if xor(offset is None, horizon is None):
        msg = "Must give both ``offset`` and ``horizon``, "
        msg = msg + "or neither."
        raise AssertionError(msg)

    # sort by ``time`,
    # group by ``time_series``,
    # only keep ``features``,
    # and make a tensor with the given ``lookback``
    # within each ``time_series``.

    # convert to datetime
    data_datetime = data[time_series + [time] + features].copy()
    data_datetime[time] = pd.to_datetime(data_datetime[time])

    grouped_df = (
        data_datetime.set_index(time_series + [time])
        .sort_index()
        .groupby(time_series)
    )

    tensors = []
    for name, group in grouped_df:
        tens = make_lookback_tensor_for_single_time_series(
            data=group,
            lookback=lookback,
            offset=offset,
            horizon=horizon,
            complete=complete,
        )
        tensors.append(tens)

    assert_true(len(tensors) > 0)

    # concatenate list of tensors into a tensor.
    # concate along the "samples" dimension.
    lookback_tensor = np.concatenate(tensors, axis=0)
    assert_true(lookback_tensor.shape[0] > 0)
    assert_true(lookback_tensor.shape[0] <= data.shape[0])
    assert_true(lookback_tensor.shape[1] == len(features))
    assert_true(lookback_tensor.shape[2] == lookback + 1)

    if lookback_tensor.shape[0] == 0:
        msg = "Somehow none of the time-series in the data contributed "
        msg = msg + "a non-empty tensor."
        raise AssertionError(msg)

    return lookback_tensor


def make_lookback_tensor_for_single_time_series(
    data, lookback, offset, horizon, complete
):
    """Converts a dataframe for a single time-series into
    a 3d tensor for deep learning.

    This method converts data for a single time-series
    from a dataframe representation to a 3d tensor representation
    that is compatible with input to a deep learning model.
    All of the data must be from the same time-series.

    Deep learning models require a tensor as input.
    Deep learning models for time-series data with multiple features
    in general require a 3d tensors where one dimension represents the
    lookback; i.e. values of the features at previous timepoints.

    The shape of the output 3d tensor is: (samples, features, lookback+1),
    where:
    - samples = the number of timepoints in the data, after dropping
        early timepoints which do not have a complete lookback
        history according to ``complete``,
        and later timepoints which do not have a full *prediction window*
        as specified by ``offset`` and ``horizon``.
    - features = the number of features in the data.

    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe of time-series data from a single time-series
        to be converted into tensor format.
        Each row corresponds to a different timepoint (sample).
        All of the samples (rows) must be from a single time-series.
        The rows must already be ordered by time.
        The rows must have an index which is the time, so that they
        can be shifted via ``data.shift()``.
        All of the columns should be features; i.e. no columns
        to represent time or the time-series grouping variables.
        Lagged feature values should not have been
        made into columns (features) yet;
        i.e. don't use ``LagAsFeatures`` before this.
    lookback : float | int
        An integer giving the number of previous timepoints to consider
        at each timepoint *t*.
        ``lookback=0`` means that only the values of the features from
        the current timepoint *t* are considered at timepoint *t*;
        `lookback=1`` means that at timepoint ``t``, the values of the features
        at timepoints ``t`` and ``t-1`` are considered; and so on.
    offset : float | int | None
        An integer. Can be positive or negative.
        Given a timepoint *t* (an integer, not timestamp),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins ``offset`` timesteps
        after the current timepoint *t*.
        For training, ``offset`` and ``horizon`` should
        be provided so that the tensor of the features (``X``)
        and the tensor of the targets (``y``) have corresponding
        entries.
        For prediction on new data, ``offset`` and ``horizon``
        should probably be set to ``None``.
    horizon : float | int | None
        An integer.
        Given a timepoint *t* (an integer),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins ``offset`` timesteps
        after the current timepoint *t*.
        For training, ``offset`` and ``horizon`` should
        be provided so that the tensor of the features (``X``)
        and the tensor of the targets (``y``) have corresponding
        entries.
        For prediction on new data, ``offset`` and ``horizon``
        should probably be set to ``None``.
    complete : bool
        if ``True``, then the output tensor only contain samples
        for which all of the previous ``lookback`` samples exist
        in the input dataframe ``data``.
        If ``False``, then the size of the first dimension of the tensor
        will be exactly the number of rows in the dataframe ``data``,
        and correspond accordingly.
    Returns
    -------
    numpy.ndarray
        A 3-dimensional numpy array; i.e. a tensor.
        The shape of the output tensor is:
            ``(samples, features, lookback + 1)``,
        where:
        - ``samples`` =
            some number not greater than the number of rows
            in the data ``data``; it can be less
            if ``complete=True`` or if ``offset`` and ``horizon``
            are not ``None`.
        - ``features`` = the number of features in the data.
        - ``lookback + 1`` because ``lookback=0`` corresponds to
            only including the values of the features at the
            current timepoint *t* when creating a tensor
            for timepoint *t*.
            - ``tensor[:,:,0]`` corresponds to timepoint *t - lookback*;
            - ``tensor[:,:,1]`` corresponds to timepoint *t - lookback + 1*;
            - ...
            - ``tensor[:,:,lookback+1]`` corresponds to timepoint *t*.
    """

    # check
    assert_type(data, pd.DataFrame)

    assert_type(lookback, [float, int])
    lookback = try_to_cast_to_integer(lookback)
    assert_true(lookback >= 0)

    assert_type(offset, [float, int], allow_None=True)
    if offset is not None:
        offset = try_to_cast_to_integer(offset)
        # offset is allowed to be negative.

    assert_type(horizon, [float, int], allow_None=True)
    if horizon is not None:
        horizon = try_to_cast_to_integer(horizon)
        assert_true(horizon >= 0)

    if xor(offset is None, horizon is None):
        msg = "Must give both ``offset`` and ``horizon``, "
        msg = msg + "or neither."
        raise AssertionError(msg)

    # verify that there are enough rows to make
    # at least one valid entry.
    if offset is not None:
        if complete is True:
            min_num_samples = lookback + offset + horizon
        else:
            min_num_samples = offset + horizon
    else:
        if complete is True:
            min_num_samples = lookback + 1
        else:
            min_num_samples = 1

    if data.shape[0] < min_num_samples:
        msg = "There are not enough records in the data to arrange "
        msg = msg + "the features as a tensor with the given "
        msg = msg + "lookback, offset and horizon.\n"
        msg = msg + f"lookback = {lookback}\n"
        msg = msg + f"offset = {offset}\n"
        msg = msg + f"horizon = {horizon}\n"
        msg = msg + f"complete = {complete}\n"
        msg = msg + f"data:\n{data.__str__()}\n"
        raise AssertionError(msg)

    assert_type(complete, bool)

    # initialize tensor
    num_samples = data.shape[0]
    num_features = data.shape[1]
    lookback_tensor = np.empty(shape=(num_samples, num_features, lookback + 1))

    # fill along the lookback dimension.
    for lagx in range(lookback + 1):
        lookback_tensor[:, :, lookback - lagx] = (
            data.shift(periods=lagx).reset_index(drop=True).to_numpy()
        )

    if complete is True:
        # trim off rows which do not have a full lookback window.
        lookback_tensor = lookback_tensor[lookback:, :, :]
        assert_true(lookback_tensor.shape[0] >= 0)
        assert_true(lookback_tensor.shape[0] == num_samples - lookback)
    else:
        assert_true(lookback_tensor.size(dim=0) == num_samples)

    if offset is not None:
        # trim off rows which do not have a full prediction window.
        # Namely, we compute the left and right endpoints of the
        # *prediction window* relative to current timepoint *t*
        # and check if any of the timepoints in the data
        # do not have a full *prediction window*.

        if offset < 0:
            # The prediction window begins to the left of current
            # timepoint *t*. Therefore, the earliest timepoints
            # in time-series will not have a complete
            # *prediction window*.
            assert_true(-offset < lookback_tensor.shape[0])
            lookback_tensor = lookback_tensor[(-offset):, :, :]

        # Compute the "overhang" with respect to a current timepoint *t*.
        right_endpoint = offset + horizon - 1
        if right_endpoint > 0:
            # so the rightmost endpoint of the prediction window is
            # to the right of current timepoint *t*.
            lookback_tensor = lookback_tensor[
                0 : (lookback_tensor.shape[0] - 1 - right_endpoint), :, :
            ]

    # check
    assert_true(lookback_tensor.shape[0] > 0)
    assert_true(lookback_tensor.shape[0] <= data.shape[0])
    assert_true(lookback_tensor.shape[1] == data.shape[1])
    assert_true(lookback_tensor.shape[2] == lookback + 1)

    return lookback_tensor


def make_prediction_window_tensor_for_multiple_time_series(
    data, time_series, time, targets, lookback, offset, horizon
):
    """Converts a dataframe of target variables for multiple time-series into
    a 3d tensor for deep learning. Forecasting-like modeling tasks involve the
     prediction of target variables for a prediction window.
    This method converts data for the target variables for
    multiple time-series from a dataframe representation
    to a 3d tensor representation that is compatible with input to a deep
    learning model. Deep learning models require the ground truth values for
    the target variables to be a input as a tensor for the loss function during
     training.Deep learning models for time-series data with multiple target
     variables in general require a 3d tensors where one dimension represents
     the horizon; i.e. values of the target variables at subsequent timesteps.
    The shape of the output 3d tensor is: (samples, targets, horizon),
    where:
    - samples = the number of timepoints in the data, after dropping
        timepoints in each time-series which do not have a full lookback window
        or do not have a full prediction window;
    - targets = the number of target variables being predicted;
    - horizon = the number of timesteps ahead being predicted;
        i.e. the length of the prediction window.
    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe with the target variables for
        time-series data from possibly multiple time-series
        to be converted into tensor format.
        Each row corresponds to a different timepoint.
    time_series : str | list of str
        Columns of the time-series dataframe ``data`` which
        can be used to distinguish distinct time-series within
        the data. For example, if ``data`` contains time-series
        data from multiple patients, then perhaps
        ``time_series=['patient_id', 'visit_id']``.
    time : str
        The column in the time-series dataframe ``data`` which
        represents time. ``data`` will be sorted by the ``time``
        column in order to create the lagged features for
        the tensor.
    targets : str | list of str
        The columns of the data which represent the target variable.
    lookback : float | int
        An integer giving the number of previous timepoints to consider
        at each timepoint *t*.
        ``lookback=0`` means that only the values of the features from
        the current timepoint *t* are considered at timepoint *t*;
        `lookback=1`` means that at timepoint ``t``, the values of the features
        at timepoints ``t`` and ``t-1`` are considered; and so on.
    offset : float | int
        An integer. Can be positive or negative.
        Given a timepoint *t* (an integer, not timestamp),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins ``offset`` timesteps
        after the current timepoint *t*.
    horizon : float | int
        An integer.
        Given a timepoint *t* (an integer),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins ``offset`` timesteps
        after the current timepoint *t*.
    Returns
    -------
    numpy.ndarray
        A 3-dimensional numpy array; i.e. a tensor.
        The shape of the output tensor is:
            ``(samples, targets, horizon)``,
        where:
        - ``samples`` =
            some number less than the number of
            timepoints in the data because
            only timepoint which have a full lookback window
            and a full prediction window are retained.
        - ``targets`` = the number of target variables in the data.
        - ``horizon`` = the length of the prediction window,
            where the prediction window for current timepoint *t*
            is the window of length ``horizon`` defined as:
            *[t + offset, t + offset + horizon]*
            ``tensor[:,:,0]`` corresponds to timepoint *t + offset + 0*,
            ``tensor[:,:,1]`` corresponds to timepoint *t + offset + 1*,
            ...,
            ``tensor[:,:,horizon-1]`` corresponds to timepoint *t + offset +
            horizon-1*.
    """

    # check
    assert_type(data, pd.DataFrame)

    assert_type_or_list_of_type(time_series, str)
    assert_columns_in_dataframe(data, time_series)
    time_series = maybe_make_list(time_series)

    assert_type(time, str)
    assert_columns_in_dataframe(data, time)
    if time in time_series:
        msg = "The ``time`` column should not be among the "
        msg = msg + "``time_series`` columns.\n"
        msg = msg + f"time: {time}\n"
        msg = msg + f"time_series: {time_series}"
        raise AssertionError(msg)

    assert_type_or_list_of_type(targets, str)
    targets = maybe_make_list(targets)
    assert_true(len(targets) > 0)
    assert_columns_in_dataframe(data, targets)
    assert_columns_are_numeric(data, targets)

    if time in targets:
        msg = "The ``time`` column should not be among the "
        msg = msg + "``targets`` columns.\n"
        msg = msg + f"time: {time}\n"
        msg = msg + f"targets: {targets}\n"
        raise AssertionError(msg)

    if any([z in targets for z in time_series]):
        msg = "The ``time_series`` and `targets`` columns should "
        msg = msg + "be disjoint.\n"
        msg = msg + f"time_series: {time_series}\n"
        msg = msg + f"targets: {targets}\n"
        raise AssertionError(msg)

    assert_type(lookback, [float, int])
    lookback = try_to_cast_to_integer(lookback)
    assert_true(lookback >= 0)

    assert_type(offset, [float, int])
    offset = try_to_cast_to_integer(offset)
    # offset is allowed to be negative.

    assert_type(horizon, [float, int])
    horizon = try_to_cast_to_integer(horizon)
    assert_true(horizon >= 0)

    # sort by ``timee`,
    # group by ``time_series``,
    # only keep ``targets``
    # and make a tensor for the subset of timepoints
    # which whose full lookback and prediction windows
    # are contained within the available timepoints
    # for the time-series
    # within each ``time_series``.

    # convert to datetime
    data_datetime = data[time_series + [time] + targets].copy()
    data_datetime[time] = pd.to_datetime(data_datetime[time])

    grouped_df = (
        data_datetime.set_index(time_series + [time])
        .sort_index()
        .groupby(time_series)
    )

    tensors = []
    for name, group in grouped_df:
        tens = make_prediction_window_tensor_for_single_time_series(
            data=group,
            lookback=lookback,
            offset=offset,
            horizon=horizon,
        )
        tensors.append(tens)

    assert_true(len(tensors) > 0)

    # concatenate list of tensors into a tensor.
    # concate along the "samples" dimension.
    prediction_window_tensor = np.concatenate(tensors, axis=0)
    assert_true(prediction_window_tensor.shape[0] > 0)
    assert_true(prediction_window_tensor.shape[0] <= data.shape[0])
    assert_true(prediction_window_tensor.shape[1] == len(targets))
    assert_true(prediction_window_tensor.shape[2] == horizon)

    return prediction_window_tensor


def make_prediction_window_tensor_for_single_time_series(
    data, lookback, offset, horizon
):
    """Converts a dataframe of target variables for a single time-series into a
     3d tensor for deep learning. Forecasting-like modeling tasks involve the
     prediction of target variables for a prediction window.
    This method converts data for the target variables for a
    single time-series from a dataframe representation
    to a 3d tensor representation that is compatible with input to a deep
    learning model. All of the data must be from the same time-series.
    Deep learning models require the ground truth for the target variables
    to be a input as a tensor for
    the loss function during training.
    Deep learning models for time-series data with multiple target variables
    in general require a 3d tensors where one dimension represents the horizon;
    i.e. values of the target variables at subsequent timesteps.
    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe of the target variables for
        time-series data from a single time-series
        to be converted into tensor format.
        Each row corresponds to a different timepoint (sample).
        All of the samples (rows) must be from a single time-series.
        The rows must already be ordered by time.
        The rows must have an index which is the time, so that they
        can be shifted via ``data.shift()``.
        All of the columns should be target variables; i.e. no columns
        to represent time or the time-series grouping variables.
    lookback : float | int
        An integer giving the number of previous timepoints to consider
        at each timepoint *t*.
        ``lookback=0`` means that only the values of the features from
        the current timepoint *t* are considered at timepoint *t*;
        `lookback=1`` means that at timepoint ``t``, the values of the features
        at timepoints ``t`` and ``t-1`` are considered; and so on.
    offset : float | int
        An integer. Can be positive or negative.
        Given a timepoint *t* (an integer, not timestamp),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins ``offset`` timesteps
        after the current timepoint *t*.
    horizon : float | int
        An integer.
        Given a timepoint *t* (an integer),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins ``offset`` timesteps
        after the current timepoint *t*.
    Returns
    -------
    numpy.ndarray
        A 3-dimensional numpy array; i.e. a tensor.
        The shape of the output tensor is:
            ``(samples, targets, horizon)``,
        where:
        - ``samples`` =
            some number less than the number of
            timepoints in the data because
            only timepoint which have a full lookback window
            and a full prediction window are retained.
        - ``targets`` = the number of target variables in the data.
        - ``horizon`` = the length of the prediction window,
            where the prediction window for current timepoint *t*
            is the window of length ``horizon`` defined as:
            *[t + offset, t + offset + horizon]*
            ``tensor[:,:,0]`` corresponds to timepoint *t + offset + 0*,
            ``tensor[:,:,1]`` corresponds to timepoint *t + offset + 1*,
            ...,
            ``tensor[:,:,horizon-1]`` corresponds to timepoint *t + offset +
             horizon-1*.
    """

    # check
    assert_type(data, pd.DataFrame)

    assert_type(lookback, [float, int])
    lookback = try_to_cast_to_integer(lookback)
    assert_true(lookback >= 0)

    assert_type(offset, [float, int])
    offset = try_to_cast_to_integer(offset)
    # offset is allowed to be negative.

    assert_type(horizon, [float, int])
    horizon = try_to_cast_to_integer(horizon)
    assert_true(horizon >= 0)

    # verify that there are enough rows to make
    # at least one valid entry.
    min_num_samples = lookback + offset + horizon
    if data.shape[0] < min_num_samples:
        msg = "There are not enough records in the data to arrange "
        msg = msg + "the target variables as a tensor with the given "
        msg = msg + "lookback, offset and horizon.\n"
        msg = msg + f"lookback = {lookback}\n"
        msg = msg + f"offset = {offset}\n"
        msg = msg + f"horizon = {horizon}\n"
        msg = msg + f"data:\n{data.__str__()}\n"
        raise AssertionError(msg)

    # initialize multidimensional array
    num_samples = data.shape[0]
    num_targets = data.shape[1]
    prediction_window_tensor = np.empty(
        shape=(num_samples, num_targets, horizon)
    )

    # fill along the horizon dimension.
    for futurex in range(horizon):
        prediction_window_tensor[:, :, futurex] = (
            data.shift(periods=-(offset + futurex))
            .reset_index(drop=True)
            .to_numpy()
        )

    # trim off rows which do not have a full lookback window.
    prediction_window_tensor = prediction_window_tensor[lookback:, :, :]
    assert_true(prediction_window_tensor.shape[0] >= 0)
    assert_true(prediction_window_tensor.shape[0] == num_samples - lookback)

    # trim off rows which do not have a full prediction window.
    # Namely, we compute the left and right endpoints of the
    # *prediction window* relative to current timepoint *t*
    # and check if any of the timepoints in the data
    # do not have a full *prediction window*.

    if offset < 0:
        # The prediction window begins to the left of current
        # timepoint *t*. Therefore, the earliest timepoints
        # in time-series will not have a complete
        # *prediction window*.
        assert_true(-offset < prediction_window_tensor.shape[0])
        prediction_window_tensor = prediction_window_tensor[(-offset):, :, :]

    # Compute the "overhang" with respect to a current timepoint *t*.
    right_endpoint = offset + horizon - 1
    if right_endpoint > 0:
        # so the rightmost endpoint of the prediction window is
        # to the right of current timepoint *t*.
        prediction_window_tensor = prediction_window_tensor[
            0 : (prediction_window_tensor.shape[0] - 1 - right_endpoint), :, :
        ]

    # check
    assert_true(prediction_window_tensor.shape[0] > 0)
    assert_true(prediction_window_tensor.shape[0] <= num_samples)
    assert_true(prediction_window_tensor.shape[1] == num_targets)
    assert_true(prediction_window_tensor.shape[2] == horizon)

    return prediction_window_tensor


def get_mapping_between_dataframe_and_lookback_tensor(
    data, time_series, time, lookback, offset, horizon, complete
):
    """Get mapping between input data and a lookback tensor.

    Suppose that you have a dataframe ``D`` and you create a lookback
    tensor ``L`` from that dataframe using the function
    ``make_lookback_tensor_for_multiple_time_series()``.

    Because of the lookback, offset, and horizon requirements, the
    number of "samples" (size of the first dimension) of ``L`` will
    in general be less than the number of rows of ``D``.
    It is therefore not straightforward to get the correspondence
    between the rows of ``D`` and the first dimension (number of "samples")
    of ``L``.

    This method will return the subset of ``D`` which corresponds row-by-row
    with the lookback tensor ``L`` in the first dimension of ``L``
    (the number of "samples"), and also the corresponding indeces of ``D``
    to obtain the subset.

    How it works
    ------------
    This method will apply the function
    ``make_lookback_tensor_for_multiple_time_series()``
    to a dataframe that is just like ``D`` except that the "features"
    are the row numbers of ``D``; any ``pandas`` row index will be ignored,
    and worse, dropped.

    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe from which a tensor was made subject to
        lookback, offset, and horizon requirements.
    time_series : str | list of str | Selector expression
        The names of the columns to group by in order
        to distinguish distinct time series
        within the data.
        For example, if the data is comprised of
        multiple time-series from multiple patients,
        we might have
        ``time_series=['patient_id', 'visit_id']``.
        The column which representds time should not be
        included here.
    time : str
        The column which represents time.
        The data will be sorted in ascending order by this column
        prior to being input to the model.
        Should not be included in the ``time_series`` column.
    lookback : float | int | IntegerDial
        The number of previous timesteps to consider when
        making a prediction at timepoint *t*.
        For example: if ``lookback=2``,
        then data from three timepoints
        will be considered by the model to make a prediction
        at timepoint *t*: *t*, *t-1*, and *t-2*.
        Don't use ``LagAsFeatures`` step before this;
        the model internally takes care of making the
        previous steps as features using ``lookback``.
    offset : float | int
        An integer.
        Given a timepoint *t* (an integer),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins ``offset`` timesteps
        after the current timepoint *t*.
    horizon : float | int
        An integer.
        Given a timepoint *t* (an integer),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins``offset`` timesteps
        after the current timepoint *t*.
    complete : bool
        if ``True``, then the output tensor only contain samples
        for which all of the previous ``lookback`` samples exist
        in the input dataframe ``data``.
        If ``False``, then the size of the first dimension of the tensor
        will be exactly the number of rows in the dataframe ``data``,
        and correspond accordingly.

    Returns
    -------
    tuple
        A tuple of length 2 with the following entries:
        - 0: a ``pandas.DataFrame`` which is a subset of ``data``
            in terms of both rows and columns:
            only the ``time_series`` and ``time`` columns
            are retained, and only the rows of ``data``
            which satisfy the ``lookback``, ``offset``, and ``horizon``
            requirements are retained; i.e. the rows of the subset
            correspond in the first dimension of the other tensor.
        - 1: a ``numpy`` array whose length is the number of rows
            in the subset of ``data`` returned in the first entry of this
            output tuple, and the values of which are the indeces of the
            input dataframe ``data`` which correspond to the row of
            the subset of ``data`` in the first entry of this output tuple.
            If the subset dataframe returned in the first entry is ``D``,
            and this index array is ``a``, then:
            ``D = data.iloc[a, :]``
            where ``data`` is the input argument to this function.
    """

    # check
    assert_type(data, pd.DataFrame)
    assert_type_or_list_of_type(time_series, str)
    assert_type(time, str)
    assert_columns_in_dataframe(data, time_series)
    assert_columns_in_dataframe(data, time)

    time_series = maybe_make_list(time_series)

    # get index mapping between rows of input ``data`` and the prediction
    # tensor.
    index_df = data[time_series + [time]].reset_index(drop=True).copy()
    index_colname = make_safe_column_name(data, "_row_index")
    index_df[index_colname] = np.arange(data.shape[0])

    # if we build a lookback tensor with the same lookback, offset, horizon
    # requirements as used elsewhere, then the indices
    # are guaranteed to match the other operation that created the other
    # lookback tensor.
    index_tensor = make_lookback_tensor_for_multiple_time_series(
        data=index_df,
        time_series=time_series,
        time=time,
        features=index_colname,
        lookback=lookback,
        offset=offset,
        horizon=horizon,
        complete=complete,
    )

    # read off the row numbers in the last index of the
    # dimension representing lookback.
    # These are the row numbers for current timepoint *t*
    # from which the lookback tensor was built.
    indeces = index_tensor[:, 0, lookback].copy()
    indeces = indeces.astype(int)

    # get the subset of the input data.
    out_df = data[time_series + [time]].reset_index(drop=True)
    out_df = out_df.iloc[indeces, :].reset_index(drop=True)
    assert_true(out_df.shape[0] == index_tensor.shape[0])

    out = (out_df, indeces)
    return out


def get_mapping_between_dataframe_and_prediction_window_tensor(
    data,
    time_series,
    time,
    lookback,
    offset,
    horizon,
):
    """Get mapping between input data and a prediction window tensor.
    Suppose that you have a dataframe ``D`` and you create a
    prediction window tensor ``P`` from that dataframe using the function
    ``make_predicton_window_tensor_for_multiple_time_series()``.
    Because of the lookback, offset, and horizon requirements, the
    number of "samples" (size of the first dimension) of ``P`` will
    in general be less than the number of rows of ``D``.
    It is therefore not straightforward to get the correspondence
    between the rows of ``D`` and the first dimension (number of "samples")
    of ``P``.
    This method will return the subset of ``D`` which corresponds row-by-row
    with the prediction window tensor ``P`` in the first dimension of ``P``
    (the number of "samples"), and also the corresponding indeces of ``D``
    to obtain the subset.
    How it works
    ------------
    This method will apply the function
    ``make_prediction_window_tensor_for_multiple_time_series()``
    to a dataframe that is just like ``D`` except that the "targets"
    are the row numbers of ``D``; any ``pandas`` row index will be ignored,
    and worse, dropped.
    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe from which a tensor was made subject to
        lookback, offset, and horizon requirements.
    time_series : str | list of str | Selector expression
        The names of the columns to group by in order
        to distinguish distinct time series
        within the data.
        For example, if the data is comprised of
        multiple time-series from multiple patients,
        we might have
        ``time_series=['patient_id', 'visit_id']``.
        The column which representds time should not be
        included here.
    time : str
        The column which represents time.
        The data will be sorted in ascending order by this column
        prior to being input to the model.
        Should not be included in the ``time_series`` column.
    lookback : float | int | IntegerDial
        The number of previous timesteps to consider when
        making a prediction at timepoint *t*.
        For example: if ``lookback=2``,
        then data from three timepoints
        will be considered by the model to make a prediction
        at timepoint *t*: *t*, *t-1*, and *t-2*.
        Don't use ``LagAsFeatures`` step before this;
        the model internally takes care of making the
        previous steps as features using ``lookback``.
    offset : float | int
        An integer.
        Given a timepoint *t* (an integer),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins ``offset`` timesteps
        after the current timepoint *t*.
    horizon : float | int
        An integer.
        Given a timepoint *t* (an integer),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins``offset`` timesteps
        after the current timepoint *t*.
    Returns
    -------
    tuple
        A tuple of length 2 with the following entries:
        - 0: a ``pandas.DataFrame`` which is a subset of ``data``
            in terms of both rows and columns:
            only the ``time_series`` and ``time`` columns
            are retained, and only the rows of ``data``
            which satisfy the ``lookback``, ``offset``, and ``horizon``
            requirements are retained; i.e. the rows of the subset
            correspond in the first dimension of the other tensor.
        - 1: a ``numpy`` array whose length is the number of rows
            in the subset of ``data`` returned in the first entry of this
            output tuple, and the values of which are the indeces of the
            input dataframe ``data`` which correspond to the row of
            the subset of ``data`` in the first entry of this output tuple.
            If the subset dataframe returned in the first entry is ``D``,
            and this index array is ``a``, then:
            ``D = data.iloc[a, :]``
            where ``data`` is the input argument to this function.
    """

    # check
    assert_type(data, pd.DataFrame)
    assert_type_or_list_of_type(time_series, str)
    assert_type(time, str)
    assert_columns_in_dataframe(data, time_series)
    assert_columns_in_dataframe(data, time)

    time_series = maybe_make_list(time_series)

    # get index mapping between rows of input ``data`` and the prediction
    # tensor.
    index_df = data[time_series + [time]].reset_index(drop=True).copy()
    index_colname = make_safe_column_name(data, "_row_index")
    index_df[index_colname] = np.arange(data.shape[0])

    # if we build a prediction window tensor with the same lookback, offset,
    # horizon
    # requirements as used to elsewhere, then the indices
    # are guaranteed to match the other operation that created the other
    # predicton window tensor.
    index_tensor = make_prediction_window_tensor_for_multiple_time_series(
        data=index_df,
        time_series=time_series,
        time=time,
        targets=index_colname,
        lookback=lookback,
        offset=offset,
        horizon=horizon,
    )

    indeces = index_tensor[:, 0, 0].copy()
    indeces = indeces.astype(int)

    # get the subset of the input data.
    out_df = data[time_series + [time]].reset_index(drop=True)
    out_df = out_df.iloc[indeces, :].reset_index(drop=True)
    assert_true(out_df.shape[0] == index_tensor.shape[0])

    out = (out_df, indeces)
    return out


def trim_and_align_time_series_outputs_with_ground_truth(
    time_series,
    time,
    offset,
    horizon,
    model_output,
    truth,
    decision_output,
):
    """Discard entries that are not relevant for scoring model outputs and
    predictions.
    Any time-series ML model used for forecasting-like tasks
    can be formulated as follows:
    When data has arrived for current timepoint *t* and the model
    needs to make a prediction, it uses data from a *lookback window* of
    previous timepoints before *t* to make predictions for timepoints in a
    *prediction window* (usually thought of as after *t*).
    The *lookback window* and *prediction window* are defined by
    the following three properties:
    - 1. lookback (integer): the *lookback window* starts
      at timepoint *t-1* and has length *lookback*.
    - 2. offset (integer): the *prediction window* begins
      at timepoint *t + offset*.
    - 3. horizon (integer): the *prediction window* has length *horizon*.
    Therefore, the number of predictions returned by ``model.transform(X)``
    is less than the number of samples in ``X``
    because, for each distinct time-series in the ``X``,
    the model does not make predictions for samples which
    do not have a complete *lookback* window.
    It is therefore necessary to "trim" ``y`` accordingly
    before passing the model predictions (``model.transform()``)
    and the ground truth values over to the metrics for scoring.
    Here, we use the model's offset and horizon to
    rearrange the results so that the predictions
    and ground truth have the same number of rows
    and the rows are in correspondence.
    How it works
    ------------
    Consider how a time-series model fits on training data:
    1. In order to run in production, any feature engineering
      recipe for time-series data *must* create lookback samples
      (step ``CreateLookbackRecords`` or something equivalent)
      for the data that is given to it because it cannot assume
      that the full lookback samples are within the data given
      to it because there may be missing samples (i.e. absence
      of records, not just rows with missing values).
      In other words, a full lookback window must be created
      for the earliest timepoint in each distinct time-series
      within the dataset ``X``.
      Therefore, the processed data ``X`` output by the
      feature engineering recipe and input to the model is
      augmented
      with lookback records no matter what; so this ``X`` is
      actually longer than ``y`` because of these added lookback
      records.
    2. During model fitting, the model reshapes the ``X`` data into a
      tensor, in which case the artificially added lookback
      records (e.g. from ``CreateLookbackRecords``)
      do not get their own place in the "number of samples"
      dimension (canonically the first dimension, in *flamingo*)
      of the tensor representation of ``X``
      (though they do show up in the "lookback timepoints"
      dimension of the tensor, which is canonically the
      third dimension, in *flamingo*).
      Timepoints in ``X`` which do not have a complete
      prediction window do not make it into the tensor, either.
      Therefore, the "number of samples" dimension of the
      ``X`` tensor is a little bit less than the number of
      rows of ``X`` (without the added lookback records)
      because of this prediction window requirement.
    3. During model fitting, the model reshapes the
      ground truth data ``y`` into a tensor as well;
      but the requirements are different. Namely,
      entries of ``y`` which do not serve in a
      prediction window for ``X`` are dropped.
      This is actually a little tricky, since ``offset``
      can be negative. Nevertheless, the tensor
      representation of ``X`` and ``y`` have the same
      size of the first dimension ("number of samples").
    Therefore, when we want the model to make predictions
    (i.e. ``model_workflow.transform()`` or ``.predict()``,
    the number of rows in the transform/prediction output
    is the same as the number of rows as in the input ``X``
    becuase of the creation (``CreateLookbackRecords``)
    and then dropping (tensor representation in the model)
    of the artificial lookback window. In other words,
    model outputs/predictions are ready-to-go in production,
    though if the input data already had the full lookback window
    (there is no way to know this within *flamingo*), then
    whoever requested the predictions from *flamingo* must know
    to select the subset of the outputs/predictions from the
    *flamingo* model of interest, since *flamingo* will return
    predictions even for the lookback window that was provided
    to it due to the ``CreateLookbackRecords``-like step that
    is effectively required in a feature engineering recipe.
    Therefore, when we want to score model output/predictions
    here, during hyperparameter tuning or model selection,
    we have to trim off some of the records from the model
    output/predictions which do not have a full prediction
    window in ``y``, and trim off corresponding records
    from ``y``.
    Parameters
    ----------
    time_series : str | list of str | Selector expression
        The names of the columns to group by in order
        to distinguish distinct time series
        within the data.
        For example, if the data is comprised of
        multiple time-series from multiple patients,
        we might have
        ``time_series=['patient_id', 'visit_id']``.
        The column which representds time should not be
        included here.
    time : str
        The column which represents time.
        The data will be sorted in ascending order by this column
        prior to being input to the model.
        Should not be included in the ``time_series`` column.
    offset : float | int
        An integer.
        Given a timepoint *t* (an integer),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins ``offset`` timesteps
        after the current timepoint *t*.
    horizon : float | int
        An integer.
        Given a timepoint *t* (an integer),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins``offset`` timesteps
        after the current timepoint *t*.
    Returns
    -------
    tuple
        A tuple of length 3 where each entry is a
        ``pandas.DataFrame`` with the same number of rows
        and sorted by ``time_series`` and ``time``.
        The entries are:
        - 0: a subset of ``model_output``;
        - 1: a subset of ``truth``;
        - 2: a subset of ``decision__output``.
    """

    # check
    assert_type_or_list_of_type(time_series, str, allow_None=True)
    assert_type(time, str)
    assert_type(offset, [float, int])
    assert_type(horizon, [float, int])
    assert_type(model_output, pd.DataFrame)
    assert_type(truth, pd.DataFrame, allow_None=True)
    assert_type(decision_output, pd.DataFrame)

    assert_true(model_output.shape[0] == decision_output.shape[0])

    if truth is not None:
        assert_true(model_output.shape[0] == truth.shape[0])

    time_series = maybe_make_list(time_series)

    if truth is not None:
        # get indeces between input data and outputs.
        (
            _,
            ix_truth,
        ) = get_mapping_between_dataframe_and_prediction_window_tensor(
            data=truth,
            time_series=time_series,
            time=time,
            lookback=0,  # ``lookback=0`` because no lookback records
            # were added to ``y``, even though they were for ``X``.
            offset=offset,
            horizon=horizon,
        )

    (
        _,
        ix_model_output,
    ) = get_mapping_between_dataframe_and_lookback_tensor(
        data=model_output,
        time_series=time_series,
        time=time,
        lookback=0,  # ``lookback=0`` because lookback records
        # were already removed.
        offset=offset,
        horizon=horizon,
        complete=True,
    )

    (
        _,
        ix_decision_output,
    ) = get_mapping_between_dataframe_and_lookback_tensor(
        data=decision_output,
        time_series=time_series,
        time=time,
        lookback=0,  # ``lookback=0`` because lookback records
        # were already removed.
        offset=offset,
        horizon=horizon,
        complete=True,
    )

    # get the same subset of rows on all outputs
    if truth is not None:
        truth = truth.iloc[ix_truth, :].reset_index(drop=True)

    model_output = model_output.iloc[ix_model_output, :].reset_index(drop=True)
    decision_output = decision_output.iloc[ix_decision_output, :].reset_index(
        drop=True
    )

    # sort so that they line up in time.
    grouping_cols = [time]
    if time_series is not None:
        grouping_cols = time_series + grouping_cols

    if truth is not None:
        truth = truth.sort_values(grouping_cols)

    model_output = model_output.sort_values(grouping_cols)
    decision_output = decision_output.sort_values(grouping_cols)

    if truth is not None:
        assert_true(truth.shape[0] == model_output.shape[0])

    assert_true(decision_output.shape[0] == model_output.shape[0])

    # return
    out = (model_output, truth, decision_output)
    return out


def make_lookback_window_feature_matrix(
    data, time, time_series, features, lookback, offset, horizon
):
    """Make a matrix of lagged features for a time-series problem.

    The input data ``X`` for time-series problems involves
    features and the value of the features for previous timepoints
    in addition to the current timepoint *t*.

    To adapt tabular models for use in time-series tasks,
    it is necessary to create a matrix (2d) which has
    the features for the current timepoint *t* and the
    features for timepoints in the *lookback window* for timepoint *t*
    in the columns.

    This function creates such a matrix, appropriately dropping any
    records which do not have a complete *lookback window* or do not have
    a complete *prediction window*.

    Parameters
    ----------
    data : pandas.DataFrame
        The data with features, time-series, and time columns.
        Almost always this represents *X*.
    time : str | list of str
        The column in the time-series dataframe ``data`` which
        represents time. ``data`` will be sorted by the ``time``
        column in order to create the lagged features for
        the tensor.
    time_series : str | list of str
        Columns of the time-series dataframe ``data`` which
        can be used to distinguish distinct time-series within
        the data. For example, if ``data`` contains time-series
        data from multiple patients, then perhaps
        ``time_series=['patient_id', 'visit_id']``.
    features : str | list of str
        The columns of the dataframe ``data`` which contain
        features. All other columns will be ignored, except for the
        columns in ``time_series`` and ``time`` which are used to arrange
        the tensor and identify appropriate lagged timepoints.
    lookback : float | int
        An integer giving the number of previous timepoints to consider
        at each timepoint *t*.
        ``lookback=0`` means that only the values of the features from
        the current timepoint *t* are considered at timepoint *t*;
        `lookback=1`` means that at timepoint ``t``, the values of the features
        at timepoints ``t`` and ``t-1`` are considered; and so on.
    offset : float | int | None
        An integer. Can be positive or negative.
        Given a timepoint *t* (an integer, not timestamp),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins ``offset`` timesteps
        after the current timepoint *t*.
        For training, ``offset`` and ``horizon`` should
        be provided so that the tensor of the features (``X``)
        and the tensor of the targets (``y``) have corresponding
        entries.
        For prediction on new data, ``offset`` and ``horizon``
        should probably be set to ``None``.
    horizon : float | int | None
        An integer.
        Given a timepoint *t* (an integer),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins ``offset`` timesteps
        after the current timepoint *t*.
        For training, ``offset`` and ``horizon`` should
        be provided so that the tensor of the features (``X``)
        and the tensor of the targets (``y``) have corresponding
        entries.
        For prediction on new data, ``offset`` and ``horizon``
        should probably be set to ``None``.

    Returns
    -------
    numpy.ndarray
        A matrix wherein the columns are the features
        at current timepoint *t* and the features for *lookback*
        previous timepoints.
        The shape is: ``(samples, features * (lookback+1))``.
        The number of samples is in general ``< data.shape[0]``
        because samples which do not have a complete *lookback window*
        or do not have a complete *prediction window* are dropped.
        The order of the columns is:
        - First feature at timestep *t - lookback*
        - First feature at timestep *t - lookback + 1*
        - ...
        - First feature at timestep *t*
        - Second feature at timestep *t - lookback*
        - Second feature at timestep *t - lookback + 1*
        - ...
        - Second feature at timestep *t*
        - ...
        - Last feature at timestep *t - lookback*
        - Last feature at timestep *t - lookback + 1*
        - ...
        - Last feature at timestep *t*
    """

    # check
    assert_time_and_time_series_are_valid(data, time, time_series)
    assert_lookback_offset_horizon_are_valid(lookback, offset, horizon)

    # format
    time_series = maybe_make_list(time_series)
    time = maybe_make_list(time)

    # make tensor of features.
    # this also drops records from each time-series that do
    # not have a full lookback window or prediction window.
    columns = time_series + time + features

    tensor = make_lookback_tensor_for_multiple_time_series(
        data=data[columns],
        time_series=time_series,
        time=time[0],
        features=features,
        lookback=lookback,
        offset=offset,
        horizon=horizon,
        complete=True,
    )
    # shape = (samples, features, lookback+1)

    # reshape data to matrix with lagged features as columns.
    num_samples = tensor.shape[0]
    num_lagged_features = tensor.shape[1] * tensor.shape[2]
    feature_matrix = np.reshape(tensor, (num_samples, num_lagged_features))

    return feature_matrix


def make_X_and_y_tensors_with_row_correspondence(
    X,
    metadata_X,
    y,
    metadata_y,
    time,
    time_series,
    lookback,
    offset,
    horizon,
    drop_nans_from_y,
):
    """Arrange ``X`` and ``y`` for training in a supervised time-series model.

    Suppose we want to train a supervised model for a
    time-series prediction task
    parameterized by *lookback*, *offset*, and *horizon*, and that
    the data is generated according to a known, uniform, fixed sampling rate,
    albeit some timepoints may be missing, such as when packets of data
    from a sensor fail to arrive.

    Given the features *X* and the ground truth outcomes *y*,
    it is necessary to arrange the entries of *X* and *y*
    according to the *lookback*, *offset*, and *horizon*
    so that the DataFrames have the same length and are in
    row-to-row correspondence for input
    into the model engine during training (i.e. ``.fit()``).

    However, there may be timepoints along the time-series for which the
    ground truth outcome is unknown or not defined. Such timepoints
    may be excluded from *y* altogether, or represented with *NaN*s as
    the ground truth outcome. It is not possible to train with *NaN* values,
    and so it is necessary to exclude corresponding entries from *X*
    and *y* (accounting for *lookback*, *offset*, and *horizon*)
    during construction of the input tensors for the
    model's ``.fit()``. Similarly, when trying to construct a row-to-row
    correspondence between *X* and *y*, it is necessary to account for
    absent entries in *y*, rather than depend on data that is contiguous
    and regular in terms of the sampling rate.

    This helper function will take the features *X* and ground truth outcomes
    *y* from the training data for a supervised learning task and arrange
    the entries of each such that there is a row-to-row correspondence, as
    required by the ``.fit()`` command for many models.
    During this process, it will account for *NaN*s in *y* and altogether
    absent timepoints from *y*; i.e. the output version of *X* and *y*
    which have row-to-row correspondence are guaranteed to not have any *NaN*
    values in *y*.

    Note: this function does not check for *NaN*s in *X*, but for most cases
    you should be sure that there are no *NaN*s in *X* upon input,
    since the ``.fit()`` for many models cannot handle *NaN*s in *X*.

    Assumptions
    -----------
    - the data is generated according to a uniform, fixed sampling rate.
    - missing data is represented by *NaN*.
    - there may be missing values, but there are no missing records;
        i.e. the lag-1 difference is identical within and across all
        time-series in the dataset.
        Including step ``StrictTimeFillGaps`` in the feature engineering
        recipe and in the target generation recipe is a convenient way
        to create placeholder records between the first and last timepoints
        of each time-series to satisfy this condition.

    Parameters
    ----------
    X : pandas.DataFrame
        The training data with the features from which predictions will
        be made by the model.
        In most cases, this should not have any *NaN*s because the output
        of this function will probably go right into the ``.fit()`` of some
        model engine.
    metadata_X : pandas.DataFrame
        Metadata for ``X``.
    y : pandas.DataFrame
        The training data with the ground truth outcomes.
        *NaN* values for the ground truth columns therein will be handled
        according to ``drop_nans_from_y``.
    metadata_y : pandas.DataFrame
        Metadata for ``y``.
    time : str | list of str
        The single column of ``X`` and ``y`` which represents time.
    time_series : str | list of str | None
        The columns of ``X`` and ``y`` which distinguish time-series
        from one another.
    lookback : float | int
        The number of previous timesteps to consider when
        making a prediction at timepoint *t*.
        For example: if ``lookback=2``,
        then data from three timepoints
        will be considered by the model to make a prediction
        at timepoint *t*: *t*, *t-1*, and *t-2*.
        Don't use ``LagAsFeatures`` step before this;
        the model internally takes care of making the
        previous steps as features using ``lookback``.
    offset : float | int
        An integer.
        Given a timepoint *t* (an integer),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins ``offset`` timesteps
        after the current timepoint *t*.
    horizon : float | int
        An integer.
        Given a timepoint *t* (an integer),
        the timesteps being predicted
        are those in the window
        ``[t + offset, t + offset + horizon - 1]``;
        the length of the prediction window is ``horizon``
        and the prediction begins``offset`` timesteps
        after the current timepoint *t*.
    drop_nans_from_y : bool
        If ``True``, then if the value in any column of ``y``
        that has role ``'ground truth'`` is *NaN*, then that entire
        row of ``y`` and the corresponding row from ``X``,
        with consideration of ``lookback``, ``offset``, and ``horizon``,
        will be excluded from the output.
        If ``False``, then ground truth columns of ``y`` will not be
        checked for *NaN*s, and therefore the outputted version of ``y``
        may have *NaN*s in rows.

    Returns
    -------
    tuple
        A tuple of length 2 where each entry is a 3d tensor
        as a ``numpy.ndarray``.

        - The first entry represents a transformed version of the
            columns within ``X`` that have role ``'feature'``; it
            has shape ``(samples, features, lookback + 1)``;
        - the second entry represents a transformed version of the
            columns within ``y`` that have role ``'ground truth'``;
            it has shape ``(samples, targets, horizon)``.

        These two tensors have the same number of rows and are in
        row-to-row correspondence, in the sense that the ground truth
        outcome corresponding to the features at row *i* of ``X``
        is found at row *i* of ``y``.
        If ``drop_nans_from_y=True``, then entries of ``y`` which had *NaN*s
        and the entries of ``X`` which would have been in correspondence
        in light of *lookback*, *offset*, and *horizon* do not appear
        in the output
        The intention is that these output DataFrames should be ready
        for input as-is to the ``.fit()`` method of a model engine.
    """

    # check
    assert_metadata_is_valid(metadata_X, X)
    assert_metadata_is_valid(metadata_y, y)
    assert_time_and_time_series_are_valid(X, time, time_series)
    assert_time_and_time_series_are_valid(y, time, time_series)
    assert_lookback_offset_horizon_are_valid(lookback, offset, horizon)
    assert_type(drop_nans_from_y, bool)

    assert_uniform_sampling_rate(X, time, time_series)
    assert_uniform_sampling_rate(y, time, time_series)

    # get columns with special roles.
    features = get_column_names_with_role(X, metadata_X, "feature")
    assert_true(features is not None and len(features) > 0)

    ground_truth = get_column_names_with_role(y, metadata_y, "ground truth")
    assert_true(ground_truth is not None and len(ground_truth) > 0)

    # the special columns must be numeric
    assert_columns_are_numeric(X, features)
    assert_columns_are_numeric(y, ground_truth)

    # create lookback tensor of features from ``X``.
    X_tensor = make_lookback_tensor_for_multiple_time_series(
        data=X,
        time_series=time_series,
        time=time,
        features=features,
        lookback=lookback,
        offset=offset,
        horizon=horizon,
        complete=True,
    )
    # shape: (samples, features, lookback + 1)

    # create prediction window tensor from ``y``.
    y_tensor = make_prediction_window_tensor_for_multiple_time_series(
        data=y,
        time_series=time_series,
        time=time,
        targets=ground_truth,
        lookback=0,
        # ``lookback=0`` because no lookback records
        # were added to ``y``, even though they were for ``X``.
        offset=offset,
        horizon=horizon,
    )
    # shape = (samples, targets, horizon)

    # The ``X`` and ``y`` tensors must have the sample number of samples;
    # i.e. size of the first dimension.
    if y_tensor.shape[0] != X_tensor.shape[0]:
        msg = (
            "The tensors created from ``X`` and ``y`` according to the "
            "``lookback``, ``offset``, and ``horizon`` criteria do not have "
            "the same number of samples (first dimension size).\n"
            "This problem can arise because you:\n"
            "  * omitted steps ``CreateLookbackRecords`` "
            "and ``StrictTimeSampleFillGaps`` in the feature engineering "
            "recipe;\n"
            "  * omitted step ``StrictTimeSampleFillGaps`` from the "
            "target generation recipe;\n"
            "  * included "
            "``CreateLookbackRecords`` in the target generation recipe.\n"
            "----\n"
            "When the data generation process has a uniform sampling rate, "
            "albeit possibly with missing data (altogether absence of "
            "records), it is helpful and even necessary to include steps"
            "``StrictTimeSampleFillGaps`` and ``CreateLookbackRecords`` "
            "in the feature engineering recipe, and include "
            "``StrictTimeSampleFillGaps`` but not `CreateLookbackRecords`` "
            "in the target generation recipe.\n"
            f"Lookback tensor shape: {X_tensor.shape}\n"
            f"Prediction window tensor shape: {y_tensor.shape}\n"
        )
        raise AssertionError(msg)

    if drop_nans_from_y is True:
        # drop entries from the ``X`` and ``y`` tensors
        # where there is a *NaN* in in ``y``.

        is_nan = np.any(np.isnan(y_tensor), axis=(1, 2))
        # shape: (samples,)

        if np.any(is_nan):
            X_tensor = X_tensor[np.logical_not(is_nan), :, :]
            y_tensor = y_tensor[np.logical_not(is_nan), :, :]

    # sanity check
    assert_true(len(X_tensor.shape) == 3)
    assert_true(len(y_tensor.shape) == 3)
    assert_true(y_tensor.shape[0] == X_tensor.shape[0])
    assert_true(not bool(np.any(np.isnan(y_tensor))))

    # return
    out = (X_tensor, y_tensor)
    return out


def make_lagged_feature_matrix_from_lookback_tensor(tensor):
    """Convert a lookback tensor to a lagged feature matrix.

    Consider a time-series problem formulated by
    *lookback*, *offset*, *horizon*.
    Suppose a *lookback tensor* with shape (samples, features, lookback+1)
    has been created from the features ``X`` from which predictions
    will be made;
    e.g. via ``make_lookback_tensor_for_multiple_time_series``.

    This helper function will reshape the 3d lookback tensor
    into a 2d matrix, where the columns of the matrix are
    lagged features.

    Parameters
    ----------
    tensor : numpy.ndarray
        A lookback tensor with shape (samples, features, lookback+1).

    Returns
    -------
    numpy.ndarray
        A matrix wherein the columns are the features
        at current timepoint *t* and the features for *lookback*
        previous timepoints.
        The shape is: ``(samples, features * (lookback+1))``.
        The order of the columns is:
        - First feature at timestep *t - lookback*
        - First feature at timestep *t - lookback + 1*
        - ...
        - First feature at timestep *t*
        - Second feature at timestep *t - lookback*
        - Second feature at timestep *t - lookback + 1*
        - ...
        - Second feature at timestep *t*
        - ...
        - Last feature at timestep *t - lookback*
        - Last feature at timestep *t - lookback + 1*
        - ...
        - Last feature at timestep *t*
    """

    # check
    assert_type(tensor, np.ndarray)
    assert_true(len(tensor.shape) == 3)
    # ``tensor`` has shape = (samples, features, lookback+1)

    # reshape tensor to matrix with lagged features as columns.
    num_samples = tensor.shape[0]
    num_lagged_features = tensor.shape[1] * tensor.shape[2]
    feature_matrix = np.reshape(tensor, (num_samples, num_lagged_features))

    # sanity check
    assert_true(len(feature_matrix.shape) == 2)
    assert_true(feature_matrix.shape[0] == tensor.shape[0])

    return feature_matrix
