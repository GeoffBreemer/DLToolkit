# Common commands

## Local machine
1. Set environment variables with EC2 instance information:

	`. gb_server_info.sh`

2. Start the EC2 instance:

	`. gb_start.sh`

3. Connect to the instance:

	`. gb_connect.sh`

3. Tunnel to the instance:

	`. gb_tunnel.sh`

4. Location of the `DLToolkit` folder on the local machine:

	`cd /Users/geoff/Documents/Development/DLToolkit`

5. Copy source files TO the server (`data` and source files only):

	`. gb_copy_source.sh`

	`. gb_copy_data.sh`

	or manually:

	`scp -i $my_pem -r dltoolkit settings thesis ubuntu@$my_dns:~/dl`

6. Location of the `exchange` folder on the local machine:

	`cd /Users/geoff/Documents/Development/DLToolkit/exchange`

7. Copy files FROM the server:

	`. gb_download_output.sh`

	`. gb_download_savedmodels.sh`

	`. gb_download_source.sh`

	or manually:

	`scp -i $my_pem ubuntu@$my_dns:~/dl/thesis/* .`

8. Stop the EC2 instance:

	`. gb_stop.sh`

## Instance

1. Activate the conda environment:

	`source activate tensorflow_p36`

2. Start Jupyter Notebook:

	`jupyter notebook`

3. UNIX commands:

	- GPU info: `nvidia-smi`

	- memory info: `free -m`

	- disk info: `df -h`

# AWS setup (one-off)
1. Login to AWS using IAM via: `https://874168575858.signin.aws.amazon.com/console`

2. Create a key pair called `deep-learning` and store it on the local machine in: `~/.ssh/deep-learning.pem`

3. Setup an EC2 Security Group called `deep-learning` with inbound rules enabling SSL, SSH and Jupyter access:

	`. gb_create_sec_group.sh`

# Local machine setup (one-off)
1. Update the `~/.bash_profile` on the local machine by adding the folder containing scripts to the `$PATH` environment variable:

	`export PATH="/Users/geoff/Documents/Development/DLToolkit/scripts/:$PATH"`

2. Install AWS CLI:

  - Install the lastest PIP: `pip install awscli --upgrade --user`

  - Run `aws configure` and set:

    - IAM user's Access Key ID: `<access key id>`
    - IAM user's Secret Access Key: `<secret access key`
    - default region name: `ap-southeast-2`
    - default output format: `text`

  - Setup Command Completion:

    - Find shell location: `echo $SHELL`
    - Locate the AWS Completer: `which aws_completer`
    - Enable completer using the path found under step 2 : `complete -C '/Users/geoff/anaconda3/bin/aws_completer' aws`
    - Add the same command to `~/.bashrc`

# Create a new Deep Learning EC2 instance (one-off)
Includes a separate second EBS Volume containing all `DLToolkit` data and source files mounted as `/home/ubuntu/dl`.

1. Create a new EC2 instance

	`. gb_create_instance.sh`

2. Set environment variables to EC2 instance information:

	`. gb_server_info.sh`

3. OPTIONAL - Create the EBS volume and attach it to the instance:

	`. gb_create_ebs.sh`

4. Copy the server setup script:

	`scp -i $my_pem /Users/geoff/Documents/Development/DLToolkit/scripts/gb_setup_server.sh ubuntu@$my_dns:~`

4. Connect to the instance:

	`. gb_connect.sh`

5. Run the server setup script (takes a minute):

	- `cd ~`
	- `chmod u+x gb_setup_server.sh`
	- `. gb_setup_server.sh`

6. OPTIONAL - Check devices: `lsblk`

7. OPTIONAL - Check if the device has a file system `sudo file -s /dev/xvdf`

8. OPTIONAL - Create the file system: `sudo mkfs -t ext4 /dev/xvdf`

9. OPTIONAL - Ensure the current location is `/home/ubuntu`: `cd ~`

10. OPTIONAL - Create mount point: `sudo mkdir dl`

11. OPTIONAL - Mount the EBS volume: `sudo mount /dev/xvdf ~/dl/`

12. OPTIONAL - Enable access: `sudo chmod go+rw .`

13. Exit the instance and restart it: `. gb_stop.sh` followed by `. gb_start.sh`

# Interact with the Deep Learning instance

Typically three terminal windows will be open:

1. One executing Jupyer Notebook on the instance
2. One to open the tunnel between the local machine
3. One to copy files back and forth

Typical workflow:

1. Start the instance
2. Start Jupyter Notebook
3. Copy files to the instance
4. Fit a model, optionally create augmented data
5. Copy files to the local machine
6. Stop the instance

## 1. Start the instance

Execute: `. gb_start.sh`

## 2. Start Jupyter Notebook (on every reboot)

### On the **server**:

- Connect to the instance: `. gb_connect.sh`
- Activate the conda environment: `source activate tensorflow_p36`
- Change to the DLToolkit folder: `cd ~/dl`
- Start Jupyter Notebook: `jupyter notebook`

Do not close the terminal window or the Jupyter Notebook will stop.

### On the **local machine** using SSL:

- Tunnel to the instance: `. gb_tunnel.sh`
- Access Jupyter via a browser (ignore the warning during the first logon): `https://127.0.0.1:8157`
- Enter the SSL password, e.g.: `<password>`
- Always use kernel `conda_tensorflow_p36`, which includes Keras

Do not close the terminal window or the browser connection with the instance will be closed.

## 3. Copy files to the server (local machine to server, +/-8 minutes incl. images)
Local machine folder containing all files: `/Users/geoff/Documents/Development/DLToolkit`

1. Copy **source files** (excl. `output`, `data` and `savedmodels`): `. gb_copy_source.sh`

2. Copy **data** files (`*.jpg` only, ignores `*.h5`): `. gb_copy_data.sh`

## 4. Fit a model
Fit models by running Jupyter notebooks. Create augmented data if required by running the `thesis_augment_data.ipynb` notebook.

## 5. Download files FROM the server
Copy files from the server to the local `exchange` folder: `cd /Users/geoff/Documents/Development/DLToolkit/exchange`

1. Download **source** files only: `. gb_download_source.sh`
2. Download **output** only: `. gb_download_output.sh`
3. Download **savedmodels** only: `. gb_download_savedmodels.sh`

Existing files with the same name are overwritten. Other files are NOT deleted.

## 6. Stop the instance

Command: `. gb_stop.sh`

# Individual AWS CLI commands

## Jupyter Notebook - On the **local machine** NOT using SSL:

- Open tunnel: `ssh -L localhost:8888:localhost:8888 -i $my_pem ubuntu@$my_dns`
- Access Jupyter: copy/paste the link, e.g.: `http://localhost:8888/?token=d2080d49d9c27b4c2f0c5190900e42aa387246e30ce73111`

## Unmount and detach the EBS volume
First unmount then detach:

1. Connect to the instance: `ssh -i $my_pem ubuntu@$my_dns`
2. Unmount: `sudo umount /dev/xvdf`
3. Detach: `aws ec2 detach-volume --volume-id $my_vol`

## Obtain instance IDs and public DNS names
Use the GUI or use AWS CLI:

`aws ec2 describe-instances --output text --query 'Reservations[*].Instances[*].[PublicDnsName,ImageId,InstanceId]'`

## Obtain volume IDs, device and delete on termination information
Obtain the EBS volume ID using the GUI or use AWS CLI:

`aws ec2 describe-volumes --query 'Volumes[*].{ID:VolumeId,Device:Attachments[*].[Device],Delete:Attachments[*].[DeleteOnTermination]}'`

## Describe instances
`aws ec2 describe-instances --output table`

## Obtain Public DNS, AMI ID and Instance ID
For all instances:

`aws ec2 describe-instances --output text --query 'Reservations[*].Instances[*].[PublicDnsName,ImageId,InstanceId]'`

For a specific instance:

`aws ec2 describe-instances --instance-ids i-0e3cefe0851209bd1 --output text --query 'Reservations[*].Instances[*].[PublicDnsName,ImageId,InstanceId]'`

## Start an instance
Command: `aws ec2 start-instances --instance-ids i-0d86fdc0f57011ddb`

## Stop an instance
Command: `aws ec2 stop-instances --instance-ids i-0d86fdc0f57011ddb`

## Reboot an instance
Command: `aws ec2 reboot-instances --instance-ids i-0d86fdc0f57011ddb`

## Create a new EBS volume
`aws ec2 create-volume --size 10 --region ap-southeast-2 --availability-zone ap-southeast-2b --volume-type gp2`

## Attach an existing EBS volume to an instance
`aws ec2 attach-volume --device /dev/sdf --volume-id vol-06b98fbf3036e350d --instance-id i-0d86fdc0f57011ddb`

## Manually mount a previously mounted EBS volume
Steps required to manually mount an EBS volume mounted previously (e.g. after a reboot) and is attached to the instance:

1. Connect: `ssh -i $my_pem ubuntu@$my_dns`
2. Check devices: `lsblk`
3. Check file system is present: `sudo file -s /dev/xvdf`
4. Mount: `sudo mount /dev/xvdf ~/dl/`

The volume will NOT be mounted again after a reboot.

## Setup automatic mounting of an attached EBS volume on reboot
Steps required to ensure an attached EBS volume is mounted automatically after a reboot:

1. Set environment variables to EC2 instance information: `. gb_server_info.sh`
2. Connect: `ssh -i $my_pem ubuntu@$my_dns`
3. Create a backup of `fstab`: `sudo cp /etc/fstab /etc/fstab.orig`
4. Edit `fstab`: `sudo nano /etc/fstab`
5. Add a line to the end: `/dev/xvdf /home/ubuntu/dl ext4 defaults,nofail 0 2`
6. Check before rebooting: `sudo mount -a`
7. Exit the instance
8. Reboot the instance: `aws ec2 reboot-instances --instance-ids $my_ins`

# Errors
- Host key verification failed. -> delete IP address from ~/.ssh/known_hosts: `sudo nano ~/.ssh/known_hosts`
