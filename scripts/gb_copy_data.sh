#!/bin/bash

# Purpose: Copy data files to the EC2 instance
# 
# To make the script executable run: chmod u+x gb_copy_data.sh
#
# Execute: . /Users/geoff/Documents/Development/DLToolkit/scripts/gb_copy_data.sh
echo Setting environment variables
source gb_server_info.sh

cd /Users/geoff/Documents/Development/DLToolkit

echo Copying data files
rsync -avz -e "ssh -i $my_pem" --include='*/' --include='*.jpg' --exclude='*.h5' data ubuntu@$my_dns:~/dl
