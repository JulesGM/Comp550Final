set -e # Close immidiately if a line returns something else than 0 (aka, if there is an error)
set -u # Close immidiately if we try to access a variable that doesn't exist.

echo -e "\n###########################################################"
echo "# Installing python"
echo "###########################################################"
module purge
module refresh

module load cuda/10.0
module load cuda/10.0/cudnn/7.6
module load anaconda
source $CONDA_ACTIVATE

echo -e "\n###########################################################"
echo "# Building and activating the VENV"
echo "###########################################################"
conda activate ourenv
