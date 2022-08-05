# DeepIFC workflow for imaging flow cytometry data

DeepIFC is a convolutional neural network based method which generates fluorescent images from brightfield and darkfield images in imaging flow cytometry data.

## Converting CIF files to HDF
[Cifconvert](https://github.com/saeyslab/cifconvert) by Lippeveld et al. (2020) is used to transform datasets from CIF format to HDF format, which DeepIFC uses. 

## DeepIFC training
Inception.py and InceptionUnet.py are required to call the InceptionUnet model based on [Cahall et al. 2019 model](https://github.com/danielenricocahall/Keras-UNet)

The model is trained in model_call_inceptionunet.py with generator_multipledata.py and train_multipledatasets.py script is used, as well as some functions from functions.py.

Example run for model_call_inceptionunet.py, training DeepIFC:
```
sbatch -o ./slurm-%j_output_8filter.txt -e ./slurm-%j_errors_8filter.txt --gres=gpu --time=100:00:00 --cpus-per-task 10 --mem=20G 
train_multipledatasets.sh /path/to/folder/ 100 22 0 8 MNC inceptionunet 5
```

Structure of training script train_multipledatasets.sh:
```
FOLDER=${1}
DATAPATH=${2}
EPOCHS=${3}
BATCH_SIZE=${4}
NORMALIZE_BACKGROUND=${5}
NUM_FILTERS=${6}
DATASET=${7}
NETWORK=${8}
TRAIN=${9}
VAL=${10}
WHICH_RUN=${11}
WANTED_CHANNEL=${12}
LEARNING_RATE=${13}
LOSS=${14}
PATIENCE=${15}
```

## Testing and UMAP tools
The creation tools for UMAP, interactive UMAP, roc and auc curves and other tools for validating results are found in celltyping_fulldata.py.

Example run for testing DeepIFC and creating UMAPs:
```
srun --time=100:00:00 --gres=gpu python celltyping_originalexperiment.py --dataset WBC --normalize_background 1 --numFilters 8 
--loss binary_crossentropy --folder TESTING --hdf_file_path /path/to/file --means_available 1
```

## Requirements (Python)
Dependencies and package requirements can be found in [requirements.txt](https://github.com/timonenv/DeepIFC/blob/master/requirements.txt).

## Interactive UMAP example
An example of the interactive UMAP tool for the MNC dataset is found in [here](https://timonenv.github.io/DeepIFC/).

