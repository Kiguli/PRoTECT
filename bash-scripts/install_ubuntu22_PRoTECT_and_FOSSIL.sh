cd ..
cp fossil-main.zip ..
 rm fossil-main.zip

# Install dependencies
apt-get update
apt-get install -y python3 python3-pip curl nano unzip findutils sed

# Unzip and clean up
cd ..
unzip fossil-main.zip

# Get the current user's username
current_user=$(whoami)

# Change ownership of fossil-main directory and its contents
chown -R $current_user:$current_user fossil-main

# Append models.py content
cat ~/PRoTECT/ex/benchmarks-deterministic/FOSSIL-versions/models.py >> ~/fossil-main/experiments/benchmarks/models.py

# Install PRoTECT dependencies
cd PRoTECT
pip install -r requirements.txt

# Install prerequisites for dreal4
curl -fsSL 'https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install_prereqs.sh' | bash

# Install fossil-main
cd ../fossil-main
pip3 install .

cd ~

# Get the current user's home directory
home_directory=$(eval echo ~)

echo "export PYTHONPATH=$PYTHONPATH:${home_directory}/PRoTECT:${home_directory}/fossil-main" >> .profile
echo "export PYTHONPATH=$PYTHONPATH:${home_directory}/PRoTECT:${home_directory}/fossil-main" >> .bashrc

mkdir mosek
