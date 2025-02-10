import os
import re
import numpy as np
import pandas as pd
from sys import path
from os import listdir
from scipy import signal
from tensorflow import keras
from datetime import timedelta
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from scipy.interpolate import interp1d
from sklearn.model_selection import LeaveOneOut
from sklearn.utils.class_weight import compute_class_weight
from scipy.signal import butter, sosfreqz, sosfiltfilt, decimate, dlti, sos2zpk
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

path.append('./src')

from tensors import (
    make_lookback_tensor_for_multiple_time_series, 
    make_prediction_window_tensor_for_multiple_time_series,
)
from models import (
    encode_categorical_label_for_classification,
)

LIFEBELL_DEV_SENSORS_DIR = '/path/to/Stance/ACC/Data'


# Changing the current working directory
os.chdir(LIFEBELL_DEV_SENSORS_DIR)

# Print the current working directory
print("The Current working directory now is: {0}".format(os.getcwd()))

# ------ load data
sampling_rate = 50

# Initialize a list to store the data from each record
data = []

# List all files in the directory
files = os.listdir(format(os.getcwd()))

# Filter for files starting with "patient_" and ending with ".dat"
patient_files = [file for file in files if file.startswith('StanceData_') and file.endswith('.dat')]

# Loop through the patient files
for file in patient_files:

    # Extract the patient id from the filename
    record_name = file.split('.')[0]
    id = record_name.split('_')[1]
    
    # Load the record using wfdb.rdsamp()
    acc_signal, header = wfdb.rdsamp(record_name)

    # Convert to DataFrame
    df = pd.DataFrame(acc_signal, columns=['packet_id', 'time_index', 'x', 'y', 'z', 'label_value'])
    df['peripheral_id'] = id  # Add the patient ID to the DataFrame

    start_time = int(re.search(r'\d+', header['comments'][1]).group()) 

    # Convert time difference to milliseconds
    time_diff_ms = (1/sampling_rate) * 1000  # 1 second = 1000 milliseconds

    # Adjust custom_index by subtracting 1, then convert it to time
    df['inferred_epoch_ms'] = start_time + (df['time_index'] - 1) * time_diff_ms

    # Append the current DataFrame to the list
    data.append(df)

# Combine all the DataFrames into a single DataFrame
stance_acc_data = pd.concat(data, ignore_index=True)

stance_acc_data['inferred_timestamp_utc'] = pd.to_datetime(stance_acc_data.inferred_epoch_ms, unit='ms')

# Sort the data
stance_acc_data = stance_acc_data.sort_values(by=['peripheral_id', 'inferred_epoch_ms'], ascending=[True, True]).reset_index(drop=True)

stance_acc_data = stance_acc_data[[
    'peripheral_id', 
    'inferred_epoch_ms', 
    'inferred_timestamp_utc',
    'x', 
    'y', 
    'z', 
    'label_value']]


acc_data = stance_acc_data.copy()


# Data Processing

'''Comment out for Downsample'''

# --- design butterworth bandpass filter and apply it
# filter parameters
sampling_rate = 50
nyq = 0.5 * sampling_rate
cutoff_fs = (0.05,2.0)
filter_order = 4

nyq_cutoff = [i/nyq for i in cutoff_fs]
filter_type = 'bandpass'
bwfilter = signal.butter(
    filter_order,
    nyq_cutoff,
    btype=filter_type,
    analog=False,
    output="sos"
)

# sort the data in preparation for filtering
acc_data.sort_values(['peripheral_id', 'inferred_epoch_ms'], inplace=True)

# apply the butterworth filter
acc_data[['x_filt', 'y_filt', 'z_filt']] = np.concatenate(
    acc_data.groupby('peripheral_id')
    .apply(
        lambda x: signal.sosfiltfilt(
            bwfilter, x[['x', 'y', 'z']].values, axis=0, padlen=0
        )
    ).values
)

data = acc_data.copy()

''' Uncomment for downsample '''

# def apply_filter_and_downsample(group):
#     # Apply bandpass filter and downsample using the custom dlti instance
#     downsampled_values = decimate(
#         group[['x', 'y', 'z']].values,
#         downsample_factor,
#         ftype=designed_filter,  # Use the custom dlti instance for downsampling
#         axis=0,
#         zero_phase=True
#     )

#     # Create a new DataFrame with downsampled values and associated metadata
#     downsampled_df = pd.DataFrame(downsampled_values, columns=['x_filt', 'y_filt', 'z_filt'])
#     downsampled_df['inferred_timestamp_utc'] = group['inferred_timestamp_utc'].iloc[::downsample_factor].reset_index(drop=True)
#     downsampled_df['label_value'] = group['label_value'].iloc[::downsample_factor].reset_index(drop=True)
#     downsampled_df['peripheral_id'] = group['peripheral_id'].iloc[0]

#     return downsampled_df

# # --- design butterworth bandpass filter and apply it

# # Define the filter parameters
# sampling_rate = 50
# downsample_rate = 25
# downsample_factor = int(sampling_rate / downsample_rate)

# nyq = 0.5 * sampling_rate
# cutoff_fs = (0.05, 2.0)
# filter_order = 4

# nyq_cutoff = [i / nyq for i in cutoff_fs]
# filter_type = 'bandpass'
# sos_coefficients = butter(
#     filter_order,
#     nyq_cutoff,
#     btype=filter_type,
#     analog=False,
#     output="sos"
# )


# zeros, poles, gain = sos2zpk(sos_coefficients)

# # Create DLTI object
# designed_filter = dlti(zeros, poles, gain, dt=1/sampling_rate)

# # sort the data in preparation for filtering
# acc_data.sort_values(['peripheral_id', 'inferred_epoch_ms'], inplace=True)

# # Apply the function to each group and create a new DataFrame
# downsampled_data = acc_data.groupby('peripheral_id').apply(apply_filter_and_downsample)

# # Reset the index
# downsampled_data.reset_index(drop=True, inplace=True)


# data = downsampled_data.copy()


data.rename(columns={'label_value': 'label'}, inplace=True)

data['label'] = data['label'].astype(int)



# only use the last 20 minutes of each time series - this cuts out all of the extra sitting from the respiration protocols
# Group the data by 'id'
grouped = data.groupby('peripheral_id')

# Create an empty dataframe to store the filtered data
filtered_data = pd.DataFrame(columns=data.columns)

# Iterate over each group
for name, group in grouped:
    # Calculate the cutoff time (i.e. 20 minutes ago from the latest timestamp in the group)
    cutoff_time = group['inferred_timestamp_utc'].max() - timedelta(minutes=20)
    
    # Filter the data to only include rows within the last 25 minutes
    filtered_group = group[group['inferred_timestamp_utc'] >= cutoff_time]
    
    # Append the filtered data to the overall dataframe
    filtered_data = pd.concat([filtered_data, filtered_group])

filtered_data['time'] = pd.to_datetime(filtered_data.inferred_timestamp_utc)
filtered_data['x'] = filtered_data.x_filt.astype(float)
filtered_data['y'] = filtered_data.y_filt.astype(float)
filtered_data['z'] = filtered_data.z_filt.astype(float)
filtered_data['label'] = filtered_data.label.astype(int)
filtered_data['id'] = filtered_data.peripheral_id
data = filtered_data.drop(['peripheral_id', 'x_filt', 'y_filt', 'z_filt', 'inferred_timestamp_utc', 'inferred_epoch_ms'], axis=1)
del filtered_data
data.head()



def make_x_y_tensors(df, lookback):
    X_tensor = make_lookback_tensor_for_multiple_time_series(
        data=df[['id', 'time', 'x', 'y', 'z']],
        time_series='id',
        time='time',
        features=['x', 'y', 'z'],
        lookback=lookback,
        offset=0,
        horizon=1,
        complete=True
    ).transpose((0,2,1))

    # remove the first 250 records from the label dataframe
    y_tensor = make_prediction_window_tensor_for_multiple_time_series(
        data=df[['id', 'time', 'label']].groupby('id').nth[(lookback):].reset_index(),
        time_series='id',
        time='time',
        targets=['label'],
        lookback=0,
        offset=0,
        horizon=1
    )
    
    # Shuffle tensors to aid in training
    y_tensor = np.squeeze(y_tensor, axis=2)
    indices = list(range(len(X_tensor)))

    X_tensor = X_tensor[indices]
    y_tensor = y_tensor[indices]

    return X_tensor, y_tensor



# Model Training


#input size in second
input_size = 5
window_size = sampling_rate * input_size


# Perform LOO cross-validation
loo = LeaveOneOut()

# Lists to store metrics for each fold

precision_micro_list = []
recall_micro_list = []
f1_micro_list = []

precision_macro_list = []
recall_macro_list = []
f1_macro_list = []

precision_weighted_list = []
recall_weighted_list = []
f1_weighted_list = []

accuracies = []
precisions = []
recalls = []
histories = []

accuracies_test = []
conf_matrix_test = []


# Define the number of classes
num_classes = len(np.unique(data.label))


# for each fold of the LOOCV
for i, (train_index, test_index) in enumerate(loo.split(data['id'].unique())):

    test_index = test_index[0]
    test_id = data['id'].unique()[test_index]


    if test_index == 0:
        val_id = data['id'].unique()[-1]
    else:
        val_id = data['id'].unique()[test_index - 1]

    train = data[~((data['id'] == test_id) | (data['id'] == val_id))]
    test = data[data['id'] == test_id]
    val = data[data['id'] == val_id]

    # determine class weights based on magnitude of imbalance 
    # in the training set
    weights = compute_class_weight(
        'balanced',
        classes=np.unique(train.label),
        y=train.label)

    X_train, y_train = make_x_y_tensors(train, window_size)
    X_val, y_val = make_x_y_tensors(val, window_size)
    X_test, y_test = make_x_y_tensors(test, window_size)

    y_train_one_hot = to_categorical(y_train, num_classes)
    y_val_one_hot = to_categorical(y_val, num_classes)


    # Define a new 1DCNN model
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(
        filters=64, 
        kernel_size=5, 
        activation='relu',
        input_shape=(window_size + 1, 3)))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=[
            'accuracy', 
            keras.metrics.Precision(name='precision'), 
            keras.metrics.Recall(name='recall'),
            ]
        )

    # Train the model
    history = model.fit(
        X_train, 
        y_train_one_hot, 
        epochs=2, 
        batch_size=128,
        class_weight={i: weights[i] for i in range(num_classes)}, 
        verbose=1,
        validation_data=(X_val, y_val_one_hot))

    y_test_pred = model.predict(X_test)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)

    # Micro averaging
    precision_micro = precision_score(y_test, y_test_pred_classes, average='micro', zero_division=1)
    recall_micro = recall_score(y_test, y_test_pred_classes, average='micro', zero_division=1)
    f1_micro = f1_score(y_test, y_test_pred_classes, average='micro', zero_division=1)

    precision_micro_list.append(precision_micro)
    recall_micro_list.append(recall_micro)
    f1_micro_list.append(f1_micro)

    # Macro averaging
    precision_macro = precision_score(y_test, y_test_pred_classes, average='macro', zero_division=1)
    recall_macro = recall_score(y_test, y_test_pred_classes, average='macro', zero_division=1)
    f1_macro = f1_score(y_test, y_test_pred_classes, average='macro', zero_division=1)

    precision_macro_list.append(precision_macro)
    recall_macro_list.append(recall_macro)
    f1_macro_list.append(f1_macro)

    # Weighted averaging
    precision_weighted = precision_score(y_test, y_test_pred_classes, average='weighted', zero_division=1)
    recall_weighted = recall_score(y_test, y_test_pred_classes, average='weighted', zero_division=1)
    f1_weighted = f1_score(y_test, y_test_pred_classes, average='weighted', zero_division=1)

    precision_weighted_list.append(precision_weighted)
    recall_weighted_list.append(recall_weighted)
    f1_weighted_list.append(f1_weighted)

    accuracy = accuracy_score(y_test, y_test_pred_classes)
    conf_matrix = confusion_matrix(y_test, y_test_pred_classes, labels=range(num_classes))


    accuracies_test.append(accuracy)
    conf_matrix_test.append(conf_matrix)

    # Save the model for this fold
    folder_path = f"{input_size}s_{sampling_rate}Hz__Models"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    model_filename = f"{input_size}s_{sampling_rate}Hz__Models/Posture_Detection_fold_{i}"
    model.save(model_filename)
    
    # Store the accuracy, precision, and recall for each fold
    for key, value in history.history.items():
        if 'val_accuracy' in key:
            accuracies.append(value[-1])
        elif 'val_precision' in key:
            precisions.append(value[-1])
        elif 'val_recall' in key:
            recalls.append(value[-1])


# Model Results


accuracy = round(np.mean(accuracies), 2)
precision = round(np.mean(precisions), 2)
recall = round(np.mean(recalls), 2)



print("Validation Set Metrics:")
print(f"Accuracy on the validation set: {accuracy}")
print(f"Precision on the validation set: {precision}")
print(f"Recall on the validation set: {recall}")


# Calculate the average over folds

accuracy_test = round(np.mean(accuracies_test), 2)

average_precision_micro = round(np.mean(precision_micro_list), 2)
average_recall_micro = round(np.mean(recall_micro_list), 2)
average_f1_micro = round(np.mean(f1_micro_list), 2)

average_precision_macro = round(np.mean(precision_macro_list), 2)
average_recall_macro = round(np.mean(recall_macro_list), 2)
average_f1_macro = round(np.mean(f1_macro_list), 2)

average_precision_weighted = round(np.mean(precision_weighted_list), 2)
average_recall_weighted = round(np.mean(recall_weighted_list), 2)
average_f1_weighted = round(np.mean(f1_weighted_list), 2)

print("Average Test Metrics over Folds:")

print("Accuracy:", accuracy_test)

print("Micro Precision:", average_precision_micro)
print("Micro Recall:", average_recall_micro)
print("Micro F1:", average_f1_micro)

print("Macro Precision:", average_precision_macro)
print("Macro Recall:", average_recall_macro)
print("Macro F1:", average_f1_macro)

print("Weighted Precision:", average_precision_weighted)
print("Weighted Recall:", average_recall_weighted)
print("Weighted F1:", average_f1_weighted)

np.savez(f"./{input_size}s_{sampling_rate}Hz__Models/confusion_matrices.npz", *conf_matrix_test)
print("Test Set Confusion Matrix was saved.")


np.save(f'./{input_size}s_{sampling_rate}Hz__Models/Accuracy.npy', accuracies_test)
np.save(f'./{input_size}s_{sampling_rate}Hz__Models/Precision.npy', precision_weighted_list)
np.save(f'./{input_size}s_{sampling_rate}Hz__Models/Recall.npy', recall_weighted_list)
np.save(f'./{input_size}s_{sampling_rate}Hz__Models/F1.npy', f1_weighted_list)


