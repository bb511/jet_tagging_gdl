# Between forced, learned or induced symmetries: Which is better for Deep Learning based jet tagging
Code for the master thesis: Between forced, learned or induced symmetries: Which is better for Deep Learning based jet tagging

## hls4ml dataset

For training models on the hls4ml dataset

For the deepset or the vector neuron
```
python3 train.py --name vector_neuron(deepset) --data_name px_py_pz_etarel_phirel_16 --loss default
```
For the transformer model
```
python3 train_transformer.py --name transformer --data_name px_py_pz_etarel_phirel_16 --loss default
```
The first time it is used, the whole hsl4ml dataset is downloaded. Then the wanted features are selected 
you can choose among
```
px, py, pz, E, Erel, pt, ptrel, eta, etarel, etarot, phi, phirel, phirot, deltaR, costhetarel, pdgid
```
you also can add jet information for some feature selection
```
px_py_pz_jet_center, pt_eta_phi_jet_center, px_py_pz_E_jet_center
```
The feature selection is saved in the hls4ml folder along with plots of the number of the constituent and the distribution of each observables. Note that the data will be automatically robust normalized (this can be changed manually in the `import_data.py` file, by changing the `get_data_hls4ml.HLS4MLData150` call.
The code is faster the second time you use a specific feature selection as the files are already saved.

Once trained the models are saved into a specific folder of the form './train_models/{model_name}_{data_name}_{loss}/trained_models/seed42/' along with a config file.

The model specificity can be modify in the `config_equivariance.yml` file. This file is used at the training time to initialize the model and the training hyperparameters.

Running the models on a SLURM based HPC cluster can be done via the SLURM script in the `slurm_scripts` folders. Make sure to adapt them for the specificity of your cluster.

The pretrained models are saved in the `pretrain_model` folder. To test them you can use the test script with the following command
```
python3 test.py --name deepset --data_name pt_etarel_phirel_8 --pretrained True
``` 

You can train the models yourselves by using the configuration files in the `pre_config_files` folder. For this use the command 
```
python3 train.py --name vector_neuron(deepset) --data_name px_py_pz_jet_center_16 --config_file ./pre_config_files/config_vn_16.yml
```

Testing models can be done through for all the models
```
python3 test.py --name transformer --data_name px_py_pz_etarel_phirel_16 --loss default
```

The requirement for the code are put in the `equivariance_env.yml` file. Once you run
```
conda create env --file equivariance_env.yml
```
it should automatically create a environnement named equivariance_env. If this does not work here are the important requirement for the code to run
conda create -n myenv python=3.11 numpy=2.2.3 matplotlib=3.10.0 pytorch=2.6.0 pyyaml=6.0.2 pip=23.3.1
conda activate myenv
pip install torchinfo==1.8.0 deepspeed==0.16.4 scikit-learn==1.6.1 scipy==1.15.2 wget==3.2 h5py==3.13.0
```
torch
tqmd
yaml
wget
h5py
numpy
pathlib
tarfile
typing
matplotlib
sklearn
numpy
scipy
argsparse
collections
torchinfo
deepspeed
```
However I can not unsure the compatibility of the code if you install the libraries via this method

## For the JetClass dataset I will work a bit more on the code
