#!/bin/bash

# Required: a *single* EC2 instance with Name (without quotes): "deep-learning". First 
# execute gb_server_info.sh to set required environment variables.
#
# Purpose: create and attach a new EBS volume to the EC2 instance
# 
# To make the script executable run: chmod u+x gb_create_ebs.sh
#
# Execute: . /Users/geoff/Documents/Development/DLToolkit/scripts/create_ebs.sh
echo EC2 instance information:
echo $my_pem
echo $my_dns
echo $my_ins

# Create EBS volume
my_vol=$(aws ec2 create-volume --output=json --size 20 --region ap-southeast-2 --availability-zone ap-southeast-2b --volume-type gp2 | grep VolumeId | awk -F\" '{print $4}')
echo $my_vol

echo Sleeping for 10 seconds
sleep 10

# Attach it to the instance
aws ec2 attach-volume --device /dev/sdf --volume-id $my_vol --instance-id $my_ins
