#!/bin/bash

# Purpose: Copy source files FROM the EC2 instance to the exchange folder on the local
# machine
# 
# To make the script executable run: chmod u+x gb_copy_source_from.sh
#
# Execute: . /Users/geoff/Documents/Development/DLToolkit/scripts/gb_copy_source_from.sh
echo Setting environment variables
source gb_server_info.sh

cd /Users/geoff/Documents/Development/DLToolkit/exchange

echo Copying source files FROM the server
mkdir thesis
mkdir dltoolkit
mkdir settings
scp -i $my_pem -r ubuntu@$my_dns:~/dl/thesis/* ./thesis
scp -i $my_pem -r ubuntu@$my_dns:~/dl/dltoolkit/* ./dltoolkit
scp -i $my_pem -r ubuntu@$my_dns:~/dl/settings/* ./settings
