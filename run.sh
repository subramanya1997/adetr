#!/bin/bash
#SBATCH --job-name=adetr
#SBATCH -o /work/pi_hzhang2_umass_edu/snagabhushan_umass_edu/adetr/logs/sbatch_logs/sbatch_log.txt
#SBATCH --time=10:00:00
#SBATCH -c 1 # Cores
#SBATCH --mem=128GB  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH --gres gpu:2
#SBATCH -G 2  # Number of GPUs

# Help 
Help()
{
    # Display Help
    echo "This script is used to run the code on the cluster."
    echo "Syntax: ./run.sh [-h]"
    echo "options:"
    echo "h            Print this Help."
    echo "run_name     Name of the job."
    echo "mode         Mode of the job. Can be train or test."
}
# variables
RUN_NAME=default
MODE=train

# Options for the job
while getopts hrun_name: option; do
    case "${option}" in
        h) Help
            exit;;
        run_name) run_name=${OPTARG};;
        mode) MODE=${OPTARG};;
        \?) echo "Invalid option: -$OPTARG" >&2
            exit 1;;
    esac
done

PYTHON=python
VENV_NAME=adetr_venv
DATASET_DIR=data

error_exit(){
    echo "$1" 1>&2
    exit 1
}

error_clean_exit(){
    echo Try again later! Removing the virtual environment directory... 
    [ -e $VENV_NAME ] && rm -rf $VENV_NAME
    error_exit "$1" 1>&2
}

if $PYTHON -c "import sys; sys.exit(1 if sys.hexversion < 0x03000000 else 0)"; then
    VENV=venv
else
    error_exit "Python 2 is not supported. Please use Python 3."
fi

if [ ! -d $DATASET_DIR ]; then
    echo extracting dataset...
    unzip ./dataset.zip -d $DATASET_DIR || error_exit "Failed to extract dataset"
    rm -rf $DATASET_DIR/__MACOSX
    echo ""
fi

if [ ! -d $VENV_NAME ]; then
    echo Checking pip is installed
    $PYTHON -m ensurepip --default-pip >/dev/null 2>&1
    $PYTHON -m pip >/dev/null 2>&1
    if [ $? -ne 0 ]; then 
        echo pip is still not installed!...
        echo Try to install it with sudo?
        echo Run: \"sudo $PYTHON -m ensurepip --default-pip\"
        exit 1
    fi
    echo Creating python virtual environment in "$VENV_NAME/"...
    $PYTHON -m $VENV $VENV_NAME || error_exit "Failed to create virtual environment"
    source $VENV_NAME/bin/activate || error_exit "Failed to source virtual environment"
    echo Upgrading pip...
    $PYTHON -m pip install --upgrade pip
    echo Installing all pip dependency inside virtual environment...
    $PYTHON -m pip install -r requirements.txt || error_clean_exit "Something went wrong while installing pip packages"
fi

source $VENV_NAME/bin/activate || error_exit "Failed to source virtual environment (try to delete '$VENV_NAME/' and re-run)"

echo Python version:
$PYTHON -c "import sys; print(sys.version)"
echo ""
nvidia-smi || error_clean_exit "GPU not found"
echo ""

# master node
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo "MASTER_ADDR: $MASTER_ADDR"

torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 \
main.py --run_name $RUN_NAME  --config ./configs/vaw_attributes.yaml --mode train \
--load ./pretrained/pretrained_resnet101_checkpoint.pth > ./logs/$RUN_NAME/train_log.txt