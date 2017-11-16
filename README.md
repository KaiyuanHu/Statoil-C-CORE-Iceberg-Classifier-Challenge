# Statoil-C-CORE-Iceberg-Classifier-Challenge

### Required software and library

Python >= 3.5\
Tensorflow >= 1.0.0\
Keras >= 2.0.2\
numpy >= 1.12.0\
pandas >= 0.20.3\
plotly >= 2.0.15\
matplotlib = 2.0.0

### Datasets
Data fields train.json, test.json\
The data (train.json, test.json) is presented in json format. The files consist of a list of images, and for each image, you can find the following fields:

- id - the id of the image
- band_1, band_2 - the flattened image data. Each band has 75x75 pixel values in the list, so the list has 5625 elements. Note that these values are not the normal non-negative integers in image files since they have physical meanings - these are float numbers with unit being dB. Band 1 and Band 2 are signals characterized by radar backscatter produced from different polarizations at a particular incidence angle. The polarizations correspond to HH (transmit/receive horizontally) and HV (transmit horizontally and receive vertically). More background on the satellite imagery can be found here.
- inc_angle - the incidence angle of which the image was taken. Note that this field has missing data marked as "na", and those images with "na" incidence angles are all in the training data to prevent leakage.
- is_iceberg - the target variable, set to 1 if it is an iceberg, and 0 if it is a ship. This field only exists in train.json.

The dataset description stated above is from **[Kaggle](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data)**.

Training data can be download [here](https://kaggle2.blob.core.windows.net/competitions-data/kaggle/7380/train.json.7z)\
Test data can be download [here](https://kaggle2.blob.core.windows.net/competitions-data/kaggle/7380/test.json.7z)
