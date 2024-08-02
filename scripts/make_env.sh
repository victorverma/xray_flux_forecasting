#!/bin/bash

conda env create --prefix ../env/ -f ../env.yml
conda activate ../env/
python -m ipykernel install --user --name=env
conda deactivate
