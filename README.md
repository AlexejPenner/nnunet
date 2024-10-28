# ZenML NNUNET 

This project borrows some code and ideas from https://github.com/MIC-DKFZ/nnUNet/tree/master.
The data used is taken from http://medicaldecathlon.com/.

## Getting the data

Go to https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2 and
download the task(s) that are relevant to you. Unpack the files into the [data](v1data)
folder to achieve the following folder structure:

```commandline
v1data/
├── Task01_BrainTumour
├── Task02_Heart
├── Task03_Liver
├── Task04_Hippocampus
├── Task05_Prostate
├── ...
```

This represents the nnunet 1.0 dataset version

## Running the code

```commandline
zenml init
python convert_and_train.py
```