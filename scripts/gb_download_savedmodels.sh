#!/bin/bash

# Purpose: Download savemodels files FROM the EC2 instance to the exchange folder on the
# local machine
# 
# To make the script executable run: chmod u+x gb_download_savedmodels.sh
#
# Execute: . /Users/geoff/Documents/Development/DLToolkit/scripts/gb_download_savedmodels.sh
echo Setting environment variables
source gb_server_info.sh

cd /Users/geoff/Documents/Development/DLToolkit/exchange

echo Downloading savedmodels files FROM the server
mkdir savedmodels
scp -i $my_pem ubuntu@$my_dns:~/dl/savedmodels/* ./savedmodels
