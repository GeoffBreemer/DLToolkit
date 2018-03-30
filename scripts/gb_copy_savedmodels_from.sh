#!/bin/bash

# Purpose: Copy savemodels files FROM the EC2 instance to the exchange folder on the local
# machine
# 
# To make the script executable run: chmod u+x gb_copy_savedmodels_from.sh
#
# Execute: . /Users/geoff/Documents/Development/DLToolkit/scripts/gb_copy_savedmodels_from.sh
echo Setting environment variables
source gb_server_info.sh

cd /Users/geoff/Documents/Development/DLToolkit/exchange

echo Copying savedmodels files FROM the server
mkdir savedmodels
scp -i $my_pem ubuntu@$my_dns:~/dl/output/* ./savedmodels
