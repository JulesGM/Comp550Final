#!/bin/bash
# ============================================================================
# Bash script to submit multiple jobs for pretraining BERT
#
# Assume we will be running this from Comp550Final/src/
# ============================================================================
set -e # Close immidiately if a line returns an error.
set -u # Close immidiately if we try to access a variable that doesn't exist.

MODEL=lstm

declare -a gres_arr=("gpu:volta:1")

echo "$MODEL"

# Optional: message to put into the created directory
dir_message="Pre-training BERT"

# Specify output directory name and path
# Place to store the SLURM error and output, as well as logs
output_dir_name="`date +"%Y-%m-%d"`_pretrain_bert"

if [[ "$USER" == "chenant" ]] ; then
  output_dir_path="/network/tmp1/chenant/sharing/comp-550/bert-pretrain/$output_dir_name"
elif [[ "$USER" == "gagnonju" ]] ; then
  output_dir_path="/network/home/gagnonju/shared/bert_pretraining/$output_dir_name"
fi

# Specify the actual SLURM job script to submit
job_file_path="./pretraining/slurm_job_pretrain_bert.sh"


# ==
# Specify job names and characteristics
# NOTE: all arrays have to be the same length

# Job name, used to name the sub-directories and output and error files
declare -a name_arr=("$MODEL")

# The path to the pretraining data directory for each job
data_base_path="/network/home/gagnonju/shared/data/parallel_jobs_logs"
declare -a data_paths=("$data_base_path/PostCrashRun_2019-12-10_filtered-out_${MODEL}")


# Specify job partitions
declare -a part_arr=("long")

# Specify resource needs

# Specify cpu need (same for all jobs)
cpu_per_task="2"

# Specify memory (RAM) need (same for all jobs)
mem_per_job="32G"

# Specify time need (same for all jobs)
time_per_job="24:00:00"


# ============================================================================
# Below is automatically ran



# ==
# Checking that the input length is adequante (arrays are of same lengths)
if [ "${#name_arr[@]}" -ne "${#data_paths[@]}" ]; then
  echo "Array length not equal, exiting."; exit
fi
if [ "${#name_arr[@]}" -ne "${#part_arr[@]}" ]; then
  echo "Array length not equal, exiting."; exit
fi
if [ "${#name_arr[@]}" -ne "${#gres_arr[@]}" ]; then
  echo "Array length not equal, exiting."; exit
fi


#for path in "${data_paths[@]}" ; do
#  if [[ ! -f "$path/"  ]] ; then
#    echo "#############################"
#    echo "\n\"$path\" doesnt exist"
#    echo "#############################"
#    ls "$path"
#    exit
#  fi
#done

# ==
# Create output parent directory if non existant
if [ ! -d "$output_dir_path" ] ; then
  echo "Creating output directory at: $output_dir_path"
  mkdir -p $output_dir_path
  echo "Directory created at `date`" >> $output_dir_path/creation_dir.txt
  echo
fi

echo $dir_message >> $output_dir_path/creation_dir.txt


# ==
# Submit jobs
arraylength=${#name_arr[@]}
for (( i=0; i<arraylength; i++ )); do
  # ==
  # Create sub-directory for this particular pretraining dataset
  output_subdir_path="$output_dir_path/${name_arr[$i]}"
  if [ ! -d "$output_subdir_path" ] ; then
    echo "Creating output directory at: $output_subdir_path"
    mkdir -p "$output_subdir_path"
  fi

  # ==
  # Set up file names
  cur_error_file="$output_subdir_path/${name_arr[$i]}_error.txt"
  cur_out_file="$output_subdir_path/${name_arr[$i]}_out.txt"

  # ==
  # Submit job to SLURM
  sbatch --cpus-per-task=$cpu_per_task \
         --partition="${part_arr[$i]}" \
         --gres="${gres_arr[$i]}" \
         --mem=$mem_per_job \
         --time=$time_per_job \
         --output="$cur_out_file" \
         --error="$cur_error_file" \
         --export=data_dir="${data_paths[$i]}",out_dir="$output_subdir_path" \
         --job-name="${name_arr[$i]}" \
         $job_file_path

  echo

done

