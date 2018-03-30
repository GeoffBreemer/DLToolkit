#!/bin/bash

# Purpose: Start the EC2 instance with Name (without quotes): "deep-learning"
# 
# To make the script executable run: chmod u+x gb_start.sh
#
# Execute: . /Users/geoff/Documents/Development/DLToolkit/scripts/gb_start.sh
echo Setting environment variables
source gb_server_info.sh

echo Starting instance $my_ins
aws ec2 start-instances --instance-ids $my_ins

echo Pausing 15 seconds to wait for assignment of PublicDnsName
sleep 15

echo Updating environment variables
source gb_server_info.sh

echo If the PublicDnsName is not set yet run gb_server_info.sh again manually.
