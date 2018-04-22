#!/bin/bash

# Purpose: Download data files FROM the EC2 instance to the exchange folder on the local
# machine
#
# To make the script executable run: chmod u+x gb_download_data.sh
#
# Execute: . /Users/geoff/Documents/Development/DLToolkit/scripts/gb_download_data.sh
echo Setting environment variables
source gb_server_info.sh

cd /Users/geoff/Documents/Development/DLToolkit/exchange

echo Downloading data files FROM the server
mkdir data
scp -i $my_pem -r ubuntu@$my_dns:~/dl/data/* ./data
