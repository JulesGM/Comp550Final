module load anaconda
conda-activate
conda create -n ourenv python=3.7 anaconda
conda activate ourenv
conda install -c anaconda tensorflow-gpu