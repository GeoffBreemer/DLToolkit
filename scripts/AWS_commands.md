# Common commands

## Local machine
1. Set environment variables with EC2 instance information:

	`. gb_server_info.sh`

2. Start the EC2 instance:

	`aws ec2 start-instances --instance-ids $my_ins`

	or:
	
	`. gb_start.sh`
	
3. Connect to the instance:

	`ssh -i $my_pem ubuntu@$my_dns`
	
	or:
	
	`. gb_connect.sh`

3. Tunnel to the instance:

	`. gb_tunnel.sh`

4. Go to the `DLToolkit` folder on the local machine:

	`cd /Users/geoff/Documents/Development/DLToolkit`

5. Copy source files TO the server (`data` and source files only):

	`. gb_copy_source.sh`
	
	`. gb_copy_data.sh`

	or manually:

	`scp -i $my_pem -r dltoolkit settings thesis ubuntu@$my_dns:~/dl`

6. Go to the `exchange` folder on the local machine:

	`cd /Users/geoff/Documents/Development/DLToolkit/exchange`

7. Copy files FROM the server:

	`. gb_download_output.sh`
	
	`. gb_download_savedmodels.sh`
	
	`. gb_download_source.sh`

	or manually:
	
	`scp -i $my_pem ubuntu@$my_dns:~/dl/thesis/* .`

8. Stop the EC2 instance:

	`aws ec2 stop-instances --instance-ids $my_ins`

	or:
	
	`. gb_stop.sh`
	
## Instance

1. Activate the conda environment:

	`source activate tensorflow_p36`

2. Start Jupyter Notebook:

	`jupyter notebook`

# AWS setup (one-off)
1. Login to AWS using IAM via: `https://874168575858.signin.aws.amazon.com/console`

2. Create a key pair called `deep-learning` and store it on the local machine in: `~/.ssh/deep-learning.pem`

3. Setup an EC2 Security Group called `deep-learning` with two inbound rules:

	- `SSH` access for `My IP` over port `22` to enable remote access
	- `Custom TCP Rule` over port `8888` from `Anywhere` (this results in two custom rules being created) to enable access to Jupyter Notebook from a local machine

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

	Use the GUI:

	- Use `Deep Learning AMI (Ubuntu) version 6.0` from Amazon
	- Deploy to a `g3.4xlarge` or `p2.xlarge` instance
	- Set the `Name` tag to `deep-learning`
	- Use Security Group `deep-learning` (required for accessing Jupyter Notebook)
	- Use existing key pair `deep-learning`

	Or use AWS CLI:

	`. gb_create_instance.sh`

2. Set environment variables to EC2 instance information: `. gb_server_info.sh`

3. NO LONGER NEEDED: Create the EBS volume and attach it to the instance: `. gb_create_ebs.sh`

4. Connect to the instance: `ssh -i $my_pem ubuntu@$my_dns` or `. gb_connect.sh`

5. Setup an SSL certificate for Jupyter Notebook:

  - Configure SSL:
    - `mkdir ssl`
    - `cd ssl`
    - `sudo openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout "cert.key" -out "cert.pem" -batch`

  - Run iPython:
    - `ipython`
    - `from IPython.lib import passwd`
    - `passwd()`
    - Set the password, e.g.: `<password>`
    - Record the password hash, e.g.: `sha1:<password hash>`
    - `exit()`

  - Edit the Jupyter config file:
    - `nano ~/.jupyter/jupyter_notebook_config.py`
    - Paste at the end of the file:
    - `c = get_config()  
      c.NotebookApp.certfile = u'/home/ubuntu/ssl/cert.pem'
      c.NotebookApp.keyfile = u'/home/ubuntu/ssl/cert.key'
      c.IPKernelApp.pylab = 'inline'
      c.NotebookApp.ip = '*'
      c.NotebookApp.open_browser = False
      c.NotebookApp.password = '<ENTER PASSWORD HASH>'`

6. NO LONGER NEEDED: Check devices: `lsblk`

7. NO LONGER NEEDED: Check if the device has a file system `sudo file -s /dev/xvdf`

8. NO LONGER NEEDED: Create the file system: `sudo mkfs -t ext4 /dev/xvdf`

9. NO LONGER NEEDED: Ensure current location is `/home/ubuntu`: `cd ~`

10. NO LONGER NEEDED: Create mount point: `sudo mkdir dl`

11. NO LONGER NEEDED: Mount the EBS volume: `sudo mount /dev/xvdf ~/dl/`

12. Create and go to the mount point: `mkdir ~/dl` and `cd ~/dl`

13. NO LONGER NEEDED: Enable access: `sudo chmod go+rw .`

14. Create `output` subfolder: `mkdir output`

15. Create `savedmodels` subfolder: `mkdir savedmodels`

16. Install missing Python packages:

	- `source activate tensorflow_p36`
	- `pip install --upgrade pip`
	- `pip install h5py sklearn progressbar2`

17. Add the path to the `dltoolkit` source files to `PYTHONPATH` by editing the `nano ~/.bashrc` and adding: `export PYTHONPATH=/home/ubuntu/dl:$PYTHONPATH`

18. Exit the instance

19. Go to `DLToolkit` folder: `cd /Users/geoff/Documents/Development/DLToolkit`

20. Test everything by copying a file: `scp -i $my_pem README.md ubuntu@$my_dns:~/dl`

# Interact with the Deep Learning instance

Typically three terminal windows will be open:

1. One executing Jupyer Notebook on the instance
2. One to open the tunnel between the local machine
3. One to copy files back and forth

Typical workflow:

1. Start the instance
2. Start Jupyter Notebook
3. Copy files to the instance
4. Fit a model
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

- Tunnel to the instanc: `. gb_tunnel.sh`
- Access Jupyter via a browser: `https://127.0.0.1:8157`
- Enter the SSL password, e.g.: `<password>`
- Always use kernel `conda_tensorflow_p36`, which includes Keras

Do not close the terminal window or the browser connection with the instance will be closed.

## 3. Copy files to the server (local machine to server, +/-8 minutes incl. images)
Local machine folder containing all files: `/Users/geoff/Documents/Development/DLToolkit`

1. Copy **source files** (excl. `output`, `data` and `savedmodels`): `. gb_copy_source.sh`

2. Copy **data** files (`*.jpg` only, ignores `*.h5`): `. gb_copy_data.sh`

## 4. Fit a model
Fit models by running Jupyter notebooks.

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

## Manually mount a previously mounted EBS volume (optional)
Steps required to manually mount an EBS volume mounted previously (e.g. after a reboot) and is attached to the instance:

1. Connect: `ssh -i $my_pem ubuntu@$my_dns`
2. Check devices: `lsblk`
3. Check file system is present: `sudo file -s /dev/xvdf`
4. Mount: `sudo mount /dev/xvdf ~/dl/`

## Setup automatic mounting of an attached EBS volume on reboot (one-off)
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
