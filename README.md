# Intensity-free Integral-based Learning of Marked Temporal Point Processes

This is the codebase for paper [Intensity-free Integral-based Learning of Marked Temporal Point Processes].


### Directory Structure
* ```config```: This folder stores all config files to control how to create a dataloader and a model. Config files are categorized and stored by model name and dataset name. For example, all config files related to training FENN on retweet dataset are in ```config/fenn/retweet```.
* ```data```: We store datasets in this folder. 
* ```parameter_set```: If you want to use ```batch_task_worker.py``` to execute many training and evaluation tasks, you shall define them in a python file and put them in this folder. 
* ```scripts```: We store training and evaluation scripts here.
* ```src```: The source file of two IFIB models and other MTPP models.
* ```synthetic_data_gen```: This directory stores two synthetic data generators.
* ```useful_utilities```: We put the postprocessing scripts here, such as calculating the Q1, Q2, and Q3 values.
* ```log```: Generated during training if not existing. We store MTPP's training logs in this folder. Caution: log files only present after the training process completes.
* ```model```: Generated during training if not existing. This folder holds the stored checkpoints.
* ```output```: Generated during evaluation if not existing. All evaluation results go into this folder.

### Datasets:
You can download all used datasets [here](). They should be placed in folder ```data/TPP```. Please create this folder if it does not exist. You can also generate your synthetic datasets using data generators in ```synthetic_data_gen```.

#### I want to train TPP models on my datasets.
The format of the dataset used in this codebase is
```
{
    # absolute timestamp sequences.
    "time_seq": 
        {"0": [1.0,26.0,33.0,1738.0, ...],
         "1": [...], 
         "2": [...], ...,},
    # event mark sequences.
    "event":
        {"0": [1,7,9,1,2,3,3,...],
         "1": [...], 
         "2": [...], ...,},
    # expected intensity values at every event.
    # Only used while evaluating models on synthetic datasets.
    # For all real-world datasets, just fill it with 0s.
    "intensity":
        {"0": [0,0,0,0,0,0,0,...],
         "1": [...], 
         "2": [...], ...,},
    # expected NLL values at every event.
    # Only used while evaluating models on synthetic datasets.
    # For all real-world datasets, just fill it with 0s.
    "score":
        {"0": [0,0,0,0,0,0,0,...],
         "1": [...], 
         "2": [...], ...,},
}
```

Lastly, you should write the number of mark types in ```num_events.txt```.


### Instructions

#### Step 1: Create the environment.

We suggest using Anaconda 3. After creating a virtualenv and activating it, use ```conda``` to install the following packages:
```bash
# Installing essential utilities.
conda install tqdm pandas seaborn matplotlib numpy scikit-learn scipy
# Installing pytorch bulit against cudatoolkit 11.7.
# For further information about how to install pytorch, please visit https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
This codebase heavily relies on [einops](https://einops.rocks/) but until now(Aug 4, 2023), Anaconda's official channel still doesn't include this user-friendly tensor manipulation tool. So you have to install it using ```pip``` or from Anaconda's community channel ```conda-forge``` using ```conda```. I personally don't use ```conda-forge```, so we use ```pip``` here. Install ```einops``` by 
```bash
python3 -m pip install einops
```

#### Step 2: Train the model.

Before training the model, please download and place the dataset files in the correct directory. They should be in ```data/TPP```.

You have two ways to start the training process. 
1. Use ```train.py``` to start one training task. Please refers to the python files in ```parameter_set/TPP``` and guides from ```python3 train.py TPP_train --help```.

2. Use ```batch_task_worker.py``` to execute multiple training tasks one by one based on parameter definitions in ```parameter_set/TPP```. We flavor using ```batch_task_worker.py```. You can find the scripts that we use to start our experiments in ```scripts```.


#### Step 3: Evaluate the model.

After completing the model training or placing provided checkpoint in ```model```, you can evaluate these TPP models on various tasks. Like the training process, you have two ways to start the evaluation task: 1. directly use ```train.py```. Please refers to the python files in ```parameter_set/TPP``` and guides from ```python3 train.py TPP_plot --help```. 2. use ```batch_task_worker.py``` to execute multiple evaluation tasks one by one following the commands in ```parameter_set/TPP```. 


### Q&As:
1. What is ```CIFIB``` and ```IFIB```? 
```CIFIB``` is the old name of ```IFIB-N```, while ```IFIB``` refers to ```IFIB-C```.
The reason is we changed the name of two proposed model in the paper after we finished all experiments.


### LICENSE

All codes are licensed under the MIT LICENSE.


### Acknowledgement

1. The implementation of LogNormmix directly comes from Shchur's [IFL-TPP](https://github.com/shchur/ifl-tpp).
2. The synthetic data generator comes from Omi's [FullyNN](https://github.com/omitakahiro/NeuralNetworkPointProcess) codebase.
3. The implementation of SAHP is modified from the [official SAHP implementation](https://github.com/QiangAIResearcher/sahp_repo).
4. The multiprocessing and DDP codes directly come from the [NSTPP](https://github.com/facebookresearch/neural_stpp) created by Chen et al.