#!/bin/bash

# Purpose: Tunnel to the EC2 instance with Name (without quotes): "deep-learning"
# 
# To make the script executable run: chmod u+x gb_tunnel.sh
#
# Execute: . /Users/geoff/Documents/Development/DLToolkit/scripts/gb_tunnel.sh
echo Setting environment variables
source gb_server_info.sh

echo Tunnelling to instance $my_dns
ssh -i $my_pem -L 8157:127.0.0.1:8888 ubuntu@$my_dns
