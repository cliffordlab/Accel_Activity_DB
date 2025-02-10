# Activity Classification from Triaxial Accelerometry in an Ambulatory Setting

This repository contains code and data for classifying patient activity using 50Hz triaxial accelerometry sensor data collected from 23 healthy subjects performing five distinct activities: lying, sitting, standing, walking, and jogging. The data is complemented with heart rate data collected using an ambulatory device. The repository includes two classification models: a binary high/low activity classifier based on signal processing techniques and a multi-class convolutional neural network (CNN) model for classifying the five activities.

# Data

The Data folder contains the following subfolders:
- **HR_Data:** Heart rate data in WFDB format.
- **Raw_ACC_Data:** Raw accelerometer data with all activity labels in WFDB format.
- **Stance_ACC_Data:** Accelerometer data with specific labels for the five activities: lying, sitting, standing, walking, and jogging in WFDB format.
