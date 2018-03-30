#!/bin/bash

# Purpose: Create a new Deep Learning AMI (Ubuntu) Version 6.0 instance
# 
# To make the script executable run: chmod u+x gb_create_instance.sh
#
# Execute: . /Users/geoff/Documents/Development/DLToolkit/scripts/gb_create_instance.sh
# Lookup the AMI ID
my_ami=$(aws ec2 describe-images --filters "Name=name,Values=Deep Learning AMI (Ubuntu) Version 6.0" --query 'Images[*].[ImageId]')

echo Retrieving the AMI ID of "Deep Learning AMI (Ubuntu) Version 6.0":
echo $my_ami

# Create the instance
echo Creating the EC2 instance, details:
echo On t2.micro in ap-southeast-2a
aws ec2 run-instances --image-id $my_ami --instance-type t2.micro --key-name deep-learning --security-groups deep-learning --placement AvailabilityZone=ap-southeast-2a --count 1 --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=deep-learning}]' 'ResourceType=volume,Tags=[{Key=Name,Value=deep-learning}]'
