#!/bin/bash

# Purpose: Stop the EC2 instance with Name (without quotes): "deep-learning"
# 
# To make the script executable run: chmod u+x gb_stop.sh
#
# Execute: . /Users/geoff/Documents/Development/DLToolkit/scripts/gb_stop.sh
echo Setting environment variables
source gb_server_info.sh

echo Stopping instance $my_ins
aws ec2 stop-instances --instance-ids $my_ins