#!/bin/bash

set -e
echo "===== UPDATING PACKAGE LIST ====="
sudo apt-get update
echo "===== UPGRADING INSTALLED PACKAGES ====="
sudo apt-get upgrade -y
echo "===== INSTALLING PYTHON3 AND PIP ====="
sudo apt-get install -y python3 python3-pip
echo "===== INSTALLING PYTHON DEPENDENCIES ====="
pip3 install -r requirements.txt
echo "===== ALL DEPENDENCIES HAVE BEEN INSTALLED SUCCESSFULLY ====="
echo "===== PRINTING SYSTEM SPECIFICATIONS ====="
echo "===== GPU ====="
nvidia-smi
echo "===== DISK ====="
df -h
echo "===== MEMORY ====="
free -h
echo "===== SETUP COMPLETE ====="
