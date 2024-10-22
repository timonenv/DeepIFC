# DeepIFC workflow for imaging flow cytometry data

DeepIFC is a convolutional neural network based method which generates fluorescent images from brightfield and darkfield images in imaging flow cytometry data.

## Publication
The DeepIFC pre-print is available here: [Biorxiv](https://www.biorxiv.org/content/10.1101/2022.08.10.503433v1.full)

## Converting CIF files to HDF
[Cifconvert](https://github.com/saeyslab/cifconvert) by Lippeveld et al. (2020) is used to transform datasets from CIF format to HDF format, which DeepIFC uses. 

## Splitting HDF files to train, test and validation sets
[H5 split](https://github.com/timonenv/DeepIFC/blob/master/h5_split.py) is a tool for splitting, merging or filling HDF files for usage with DeepIFC. It enables the use of datasets from other studies that do not match the dimensions of the original MNC data used to train DeepIFC, for example with zero-shot learning. 

## DeepIFC training
Inception.py and InceptionUnet.py are required to call the InceptionUnet model based on [Cahall et al. 2019 model](https://github.com/danielenricocahall/Keras-UNet)

The model is trained in model_call_inceptionunet.py with generator_multipledata.py and train_multipledatasets.py script is used, as well as some functions from functions.py.

The weights for the complete model are available in [checkpoints_complete](https://github.com/timonenv/DeepIFC/tree/master/checkpoints_complete).
The weights for the best iterations for each cell type model in balanced data are available in [checkpoints_balanced](https://github.com/timonenv/DeepIFC/tree/master/checkpoints_balanced) under respective folders.

Example run for model_call_inceptionunet.py, training DeepIFC:
```
train_multipledatasets.sh /path/to/BASEPATH/ SAVEFOLDER /path/to/DATAPATH/ 100 25 1 8 MNC inceptionunet 2 0.001 binary_crossentropy 5
```
where the parameters in order are:
```
BASEPATH
SAVEFOLDER
HDF data folder
epochs
batch size
normalizing fluorescent image background (0=no, 1=yes)
filter amount
dataset name
neural network
save folder
channel number
learning rate
loss
patience
(pretrained weights)
```

## Testing and UMAP tools
The creation tools for UMAP, interactive UMAP, roc and auc curves and other tools for validating results are found in celltyping_fulldata.py.

Example run for testing DeepIFC and creating UMAPs:
```
python celltyping_originalexperiment.py --dataset WBC --normalize_background 1 --numFilters 8 --loss binary_crossentropy --folder TESTING --hdf_file_path /path/to/file --means_available 1
```
where the parameters in order are:
```
name of dataset
normalizing fluorescent image background (0=no, 1=yes)
filter amount
loss
save folder
path to HDF file
availability of means for images (0=no, 1=yes)
```
The means for all images in the test set must be calculated before cells can be typed for analysis. This is done by running the same command as for celltyping_originalexperiment.py, but changing the option for the means_available parameter to 0.


## Requirements (Python)
Full package requirements can be found in [requirements.txt](https://github.com/timonenv/DeepIFC/blob/master/requirements.txt).

Dependencies for model:
* Python 3.7.9
* tensorflow-gpu 2.4.1

Packages for visualization:
* bokeh 2.4.3
* matplotlib 3.2.2
* pandas 1.3.5
* scikit-learn 1.0.2
* seaborn 0.11.2
* umap-learn 0.5.3

## Interactive UMAP example
An example of the interactive UMAP tool for the MNC dataset is found in [here](https://timonenv.github.io/DeepIFC/). Visualized are 3600 cells from different donors. 

