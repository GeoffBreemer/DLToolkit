#!/bin/bash

# Purpose: Connect to the EC2 instance with Name (without quotes): "deep-learning"
# 
# To make the script executable run: chmod u+x gb_connect.sh
#
# Execute: . /Users/geoff/Documents/Development/DLToolkit/scripts/gb_connect.sh
echo Setting environment variables
source gb_server_info.sh

echo Connecting to instance $my_dns
ssh -i $my_pem ubuntu@$my_dns
