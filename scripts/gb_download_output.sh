#!/bin/bash

# Purpose: Download output files FROM the EC2 instance to the exchange folder on the local
# machine
# 
# To make the script executable run: chmod u+x gb_download_output.sh
#
# Execute: . /Users/geoff/Documents/Development/DLToolkit/scripts/gb_download_output.sh
echo Setting environment variables
source gb_server_info.sh

cd /Users/geoff/Documents/Development/DLToolkit/exchange

echo Downloading output files FROM the server
mkdir output
scp -i $my_pem ubuntu@$my_dns:~/dl/output/* ./output
