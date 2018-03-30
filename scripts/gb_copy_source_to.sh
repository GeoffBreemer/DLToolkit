#!/bin/bash

# Purpose: Copy source files to the EC2 instance
# 
# To make the script executable run: chmod u+x gb_copy_source_to.sh
#
# Execute: . /Users/geoff/Documents/Development/DLToolkit/scripts/gb_copy_source_to.sh
echo Setting environment variables
source gb_server_info.sh

cd /Users/geoff/Documents/Development/DLToolkit

echo Copying source files
scp -i $my_pem -r dltoolkit settings thesis ubuntu@$my_dns:~/dl
