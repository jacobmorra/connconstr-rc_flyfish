# connconstr-rc_flyfish
We build connectome-constrained models using fruit fly and zebrafish neural networks.

## Experiment Code
Each folder corresponds to an experiment from the paper. For example, Exp1 contains the code for experiment 1.

## Important Files/Folders
### Data/Models
- Each experiment folder contains data (i.e. datasets) in Data_Files and models in Model_Files folders. 

### Connectomes
- Wnorm.csv is the result of querying Janelia's EM hemibrain. This is the connectome first used in experiment 1.
- null.npy is obtained from the separate query process described in the paper. It is used for experiment 2.
- LHRdiagtol50.npy is the connectome file used for experiment 3. It applies a weight threshold of 50 (no weights < 50 synapses).

### Running code
Before you start...
1) Install Python 3.7x (i.e. conda) https://www.anaconda.com/products/distribution.
2) Install Jupyter Notebook: https://jupyter.org/install or Jupyter Lab.
3) Open Jupyter Notebook in the appropriate directory, or move into it.
4) Install additional dependencies if needed (you can use pip): 
- "dill" -> https://pypi.org/project/dill/
- "progressbar" -> https://pypi.org/project/progressbar/
- "networkx" -> https://pypi.org/project/networkx/
- "pickle" -> https://pypi.org/project/pickle5/

For any questions, please contact jacob.morra@duke.edu
