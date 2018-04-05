#!/bin/bash

# Purpose: Create a security group enabling SSL, SSH and Jupyter access
# 
# To make the script executable run: chmod u+x gb_create_sec_group.sh
#
# Execute: . /Users/geoff/Documents/Development/DLToolkit/scripts/gb_create_sec_group.sh
echo Creating the deep-learning Security Group

# Create the AWS security group
aws ec2 create-security-group --group-name deep-learning --description "Allow SSL, SSH and Jupyter access"
 
# Add the Jupyter rule
aws ec2 authorize-security-group-ingress --group-name deep-learning --protocol tcp --port 8888 --cidr 0.0.0.0/0
 
# Add the SSH rule
aws ec2 authorize-security-group-ingress --group-name deep-learning --protocol tcp --port 22 --cidr 0.0.0.0/0
 
# Add the SSL rule
aws ec2 authorize-security-group-ingress --group-name deep-learning --protocol tcp --port 443 --cidr 0.0.0.0/0
