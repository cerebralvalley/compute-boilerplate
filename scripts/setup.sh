#!/bin/bash

set -e
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y python3 python3-pip
pip3 install -r requirements.txt
echo "All dependencies have been installed successfully."
