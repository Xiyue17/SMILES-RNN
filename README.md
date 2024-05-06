# SMILES-RNN
The code is the basis of our submission (Neural network prediction model for dew point and bubble point phase equilibria behavior of binary mixtures in alcohol systems ) and is designed to utilize SMILES and thermodynamic properties to train recurrent neural networks to predict vapor-liquid equilibrium data.
# Dependencies of the model
The code is written in python. 
The implementation of the model relies on the following libraries and tools:
- python 3.6
- pandas==1.1.5
- numpy==1.19.5
- tensorflow==2.1.0
- rdkit==2021.9.4
- scikit-learn==0.24.2
- keras==2.3.1
- matplotlib==2.2.2
# smiles_onehot.py
The dictionary is created using "compound" as input and the final output is the one-hot encoding of the SMILES contained in the dataset.
