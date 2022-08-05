# DeepIFC workflow for imaging flow cytometry data

DeepIFC is a convolutional neural network based method which generates fluorescent images from brightfield and darkfield images in imaging flow cytometry data.

## Converting CIF files to HDF
[Cifconvert](https://github.com/saeyslab/cifconvert) by Lippeveld et al. (2020) is used to transform datasets from CIF format to HDF format, which DeepIFC uses. 

## DeepIFC
Inception.py and InceptionUnet.py are required to call the InceptionUnet model based on [Cahall et al. model](https://github.com/danielenricocahall/Keras-UNet)

The model is trained in model_call_inceptionunet.py with generator_multipledata.py and train_multipledatasets.py script is used, as well as some functions from functions.py.

## Testing and UMAP tools
The creation tools for UMAP, interactive UMAP, roc and auc curves and other tools for validating results are found in celltyping_fulldata.py.

## Requirements (Python)
Dependencies and package requirements can be found in [requirements.txt](https://github.com/timonenv/DeepIFC/blob/master/requirements.txt).

## Interactive UMAP example
An example of the interactive UMAP tool for the MNC dataset is found in [here](https://timonenv.github.io/DeepIFC/).

