# Activity Classification from Triaxial Accelerometry in an Ambulatory Setting

This repository contains code and data for classifying patient activity using 50Hz triaxial accelerometry sensor data and heart rate data sampled at 1Hz, collected from 23 healthy subjects performing five distinct activities: lying, sitting, standing, walking, and jogging. The data is complemented with heart rate data collected using an ambulatory device. The repository includes two classification models: a binary high/low activity classifier based on signal processing techniques; and a multi-class convolutional neural network (CNN) model for classifying the five activities.

### Data

The Data folder contains the following subfolders:
- **HR_Data:** Heart rate data in WFDB format.
- **Raw_ACC_Data:** Raw accelerometer data with all activity labels in WFDB format.
- **Stance_ACC_Data:** Accelerometer data with specific labels for the five activities: lying, sitting, standing, walking, and jogging in WFDB format.

### Sensor setup and axis orientation

- **Device & placement:** VivaLNK VV330 chest patch, placed in Lead II position.
- **Sampling:** Accelerometry at 50 Hz; heart rate at 1 Hz.
- **Axis orientation (verified on raw data):**
  - **X:** along the Lead II diagonal (right shoulder → left lower chest)
  - **Y:** anterior ↔ posterior (front ↔ back)
  - **Z:** points inward toward the chest; gravity appears as **+Z ≈ +1 g** when upright

These conventions should be followed when reproducing features, re-training the CNN model, or comparing with external accelerometry datasets.

### Codes

The code/ folder contains the following files:
- **src:** Helper functions for preprocessing and analysis.
- **Activity_Counts_Motion_Analysis_WFDB.ipynb:** Signal processing approach for activity classification.
- **Activity_Detection_Model.py:** CNN-based activity classification model.
- **Activity_Detection_Results_Analysis.ipynb:** Code to analyze and visualize the model results.

### License

The code and data are available under a BSD 3-Clause open-source license. See the LICENSE file for details.

### Citation

The paper describing the code and data is included in the Documents folder. If you use this code or data in your research, please cite the paper as follows:

**Nikookar, S., Tian, E., Hoffman, H., Parks, M., McKay, J.L., Kiarashi, Y., Thomas, T.T., Hall, A., Wright, D.W. and Clifford, G.D., Activity Classification from Triaxial Accelerometry in an Ambulatory Setting. Sensors, Under Review, January 2026.**

### Contact

For any questions or issues, feel free to contact Sepideh Nikookar at: [sepideh.nikookar@emory.edu](mailto:sepideh.nikookar@emory.edu)
