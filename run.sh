#!/bin/bash

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

cd "`dirname \"$0\"`"
if $PYTHON -c "import sys; sys.exit(1 if sys.hexversion < 0x03000000 else 0)"; then
    VENV=venv
else
    error_exit "Python 2 is not supported. Please use Python 3."
fi

if [ ! -d $DATASET_DIR ]; then
    echo extracting dataset...
    unzip dataset.zip -d $DATASET_DIR || error_exit "Failed to extract dataset"
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

nvidia-smi || echo "GPU not found"

echo ""

$PYTHON main.py --config ./configs/vaw_attributes.yaml --mode train