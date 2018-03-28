# Pre-requisites

## AWS setup
1. Login to AWS using IAM via:
https://874168575858.signin.aws.amazon.com/console

2. On AWS setup a Security Group with two inbound rules:

	- `SSH` access for `My IP` over port `22` (to enable remote access)
	- `Custom TCP Rule` over port `8888` from `Anywhere` (results in two custom rules being created, this enables access to Jupyter Notebook)

## Local machine setup
1. Update the `~./bash_profile` on the local machine to add the folder containing scripts to the `$PATH` environment variable:

	`export PATH="/Users/geoff/Documents/Development/DLToolkit/scripts/:$PATH"`

2. Install AWS CLI:

  1. Install using PIP: `pip install awscli --upgrade --user`
  2. Run `aws configure` and set:

    - IAM user's Access Key ID: `<access key id>`
    - IAM user's Secret Access Key: `<secret access key`
    - default region name: `ap-southeast-2`
    - default output format: `text`

  3. Setup Command Completion:

    - Find shell location: `echo $SHELL`
    - Locate the AWS Completer: `which aws_completer`
    - Enable completer using the path found under step 2 : `complete -C '/Users/geoff/anaconda3/bin/aws_completer' aws`
    - Add the same command to `~/.bashrc`

# Create a new Deep Learning EC2 instance
Includes a separate second EBS Volume containing `DLToolkit` data and source files mounted to `/home/ubuntu/dl`.

## 1. Initial EC2 instance setup (one-off)

1. Create a new EC2 instance using the GUI:

  - Use `Deep Learning AMI (Ubuntu)` from Amazon
  - Set the `Name` tag to `deep-learning`
  - Deploy to a `g3.4xlarge` or `p2.xlarge` instance
  - Use Security Group `deep-learning` (required for accessing Jupyter Notebook)

2. Set environment variables to EC2 instance information: `. gb_server_info.sh`

3. Create the EBS volume and attach it to the instance: `. gb_create_ebs.sh`

4. Connect to the instance: `ssh -i $my_pem ubuntu@$my_dns`

5. Setup Jupyter Notebook:

  - Setup an SSL certificate (https://docs.aws.amazon.com/dlami/latest/devguide/setup-jupyter-config.html):
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
      - `exit`

    - Edit the config file:
      - `nano ~/.jupyter/jupyter_notebook_config.py`
      - Paste at the end of the file:
      - `c = get_config()  
  c.NotebookApp.certfile = u'/home/ubuntu/ssl/cert.pem'
  c.NotebookApp.keyfile = u'/home/ubuntu/ssl/cert.key'
  c.IPKernelApp.pylab = 'inline'
  c.NotebookApp.ip = '*'  
  c.NotebookApp.open_browser = False
  c.NotebookApp.password = <ENTER PASSWORD HASH>`

6. Check devices: `lsblk`

7. Check if the device has a file system `sudo file -s /dev/xvdf`

8. Create the file system: `sudo mkfs -t ext4 /dev/xvdf`

9. Ensure current location is `/home/ubuntu`: `cd ~`

10. Create mount point: `sudo mkdir dl`

11. Mount the EBS volume: `sudo mount /dev/xvdf ~/dl/`

12. Go to the mount point: `cd dl`

13. Enable access: `sudo chmod go+rw .`

14. Create `output` subfolder: `mkdir output`

15. Create `savedmodels` subfolder: `mkdir savedmodels`

16. Go to `DLToolkit` folder: `cd /Users/geoff/Documents/Development/DLToolkit`

17. Test the volume by copying a file: `scp -i $my_pem README.md ubuntu@$my_dns:~/dl`

18. Add the path to the dltoolkit source files to `PYTHONPATH` by editing the `~/.bashrc` and adding: `export PYTHONPATH=/home/ubuntu/dl/dltoolkit:$PYTHONPATH`

19. Exit the instance

## 2. Copy all `DLToolkit` files (local machine to server, +/-8 minutes)
Copy all relevant files and subfolders:

1. On the local machine go to folder: `cd /Users/geoff/Documents/Development/DLToolkit`
2. Copy all subfolder content (excl. `output` and `savedmodels`): `scp -i $my_pem -r data dltoolkit examples_complex examples_simple settings thesis ubuntu@$my_dns:~/dl`
3. Copy all files in the root folder: `scp -i $my_pem * ubuntu@$my_dns:~/dl`


## 3. Manually mount a previously mounted EBS volume (optional)
Steps required to manually mount an EBS volume mounted previously (e.g. after a reboot) and is attached to the instance:

1. Connect: `ssh -i $my_pem ubuntu@$my_dns`
2. Check devices: `lsblk`
3. Check file system is present: `sudo file -s /dev/xvdf`
4. Mount: `sudo mount /dev/xvdf ~/dl/`


## 4. Setup automatic mounting of an attached EBS volume on reboot (one-off)
Steps required to ensure an attached EBS volume is mounted automatically after a reboot:

1. Create a backup of `fstab`: `sudo cp /etc/fstab /etc/fstab.orig`
2. Edit `fstab`: `sudo nano /etc/fstab`
3. Add a line to the end: `/dev/xvdf /home/ubuntu/dl ext4 defaults,nofail 0 2`
4. Check before rebooting: `sudo mount -a`
5. Reboot instance: `aws ec2 reboot-instances --instance-ids my_ins`


# Interact with the Deep Learning instance

## 1. Start the instance

- Obtain server information (only sets `my_ins`): `. gb_server_info.sh`
- Start the EC2 instance: `aws ec2 start-instances --instance-ids my_ins`
- Obtain server information (now also sets `my_dns`): `. gb_server_info.sh`

## 2. Setup Jupyter Notebook (on every startup)

### On the **server**:

- Connect to the instance: `ssh -i $my_pem ubuntu@$my_dns`
- Activate the conda environment: `source activate tensorflow_p36`
- Start Jupyter Notebook: `jupyter notebook`

### On the **local machine** using SSL:

- Open tunnel: `ssh -i $my_pem -L 8157:127.0.0.1:8888 ubuntu@$my_dns`
- Access Jupyter via the browser: `https://127.0.0.1:8157`
- Enter the SSL password, e.g.: `<password>`

### On the **local machine** NOT using SSL:

- Open tunnel: `ssh -L localhost:8888:localhost:8888 -i $my_pem ubuntu@$my_dns`
- Access Jupyter: copy/paste the link, e.g.: `http://localhost:8888/?token=d2080d49d9c27b4c2f0c5190900e42aa387246e30ce73111`

## 3. Unmount and detach the EBS volume
First unmount then detach:

1. Connect to the instance
2. Unmount: `sudo umount /dev/xvdf`
3. Detach: `aws ec2 detach-volume --volume-id $my_vol`

## 4. Download files from the server
Copy files from the server to a local exchange folder:

1. Go the folder where files are to be downloaded to: `cd /Users/geoff/Documents/Development/exchange`
2. Download files: `scp -i $my_pem ubuntu@$my_dns:~/dl/test.txt .`

## 5. Stop the instance

Command: `aws ec2 stop-instances --instance-ids $my_ins`


# Useful AWS CLI commands

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

# Errors
- Host key verification failed. -> delete IP address from ~/.ssh/known_hosts: `sudo nano ~/.ssh/known_hosts`
