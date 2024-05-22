cd ..
apt-get update
apt-get install -y python3 python3-pip findutils sed
pip install -r requirements.txt

cd ~

# Get the current user's home directory
home_directory=$(eval echo ~)

echo "export PYTHONPATH=$PYTHONPATH:${home_directory}/PRoTECT:${home_directory}/fossil-main" >> .profile
echo "export PYTHONPATH=$PYTHONPATH:${home_directory}/PRoTECT:${home_directory}/fossil-main" >> .bashrc

mkdir mosek
