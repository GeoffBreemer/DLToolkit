#!/bin/bash

# Required: a *single* EC2 instance with Name (without quotes): "deep-learning"
#
# Purpose: set the my_ins=InstanceID and my_dns=PublicDnsName variables using the EC2
# instance, set the path to the .pem file
#
# To make the script executable run: chmod u+x gb_server_info.sh
#
# Execute: . /Users/geoff/Documents/Development/DLToolkit/scripts/gb_server_info.sh

# Set variables
my_pem=/Users/geoff/.ssh/deep-learning.pem
my_dns=$(aws ec2 describe-instances --output text --filters "Name=tag:Name,Values=deep-learning" --query 'Reservations[*].Instances[*].[PublicDnsName]')
my_ins=$(aws ec2 describe-instances --output text --filters "Name=tag:Name,Values=deep-learning"  --query 'Reservations[*].Instances[*].[InstanceId]')

echo EC2 instance information:
echo $my_pem
echo $my_dns
echo $my_ins
