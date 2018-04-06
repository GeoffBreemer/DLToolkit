#!/bin/bash

# Purpose: Setup an SSL certificate for Jupyter Notebook on the server instance
# 
# To make the script executable run: chmod u+x gb_setup_server
#
# Execute: . gb_setup_server

# jupyter notebook --generate-config
echo Setup Jupyter SSL
key=$(python -c "from notebook.auth import passwd; print(passwd())")

cd ~
mkdir ssl
cd ssl
certdir=$(pwd)
sudo openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout gb_cert.key -out gb_cert.pem -batch

#cd ~
#sed -i "" -e $"$ a\\

sed -i "$ a\
c = get_config()\\
c.NotebookApp.certfile = u'$certdir/gb_cert.pem'\\
c.NotebookApp.keyfile = u'$certdir/gb_cert.key'\\
c.NotebookApp.ip = '*'\\
c.IPKernelApp.pylab = 'inline'\\
c.NotebookApp.open_browser = False\\
c.NotebookApp.password = u'$key'" ~/.jupyter/jupyter_notebook_config.py

echo Finished SSL setup for Jupyter

echo Create subfolders
cd ~
mkdir dl
mkdir ~/dl/output
mkdir ~/dl/output/segmentation_maps
mkdir ~/dl/savedmodels

echo Install Python packages, first activation may take some time
source activate tensorflow_p36
pip install --upgrade pip
pip install h5py sklearn progressbar2 pydot pydot_ng graphviz seaborn

echo Update PYTHONPATH, ignore the error message
sed -i "" -e $'$ a\\\n''export PYTHONPATH=/home/ubuntu/dl:$PYTHONPATH' ~/.bashrc
