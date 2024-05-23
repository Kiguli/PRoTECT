cd ..
mv fossil-main ..

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip curl nano unzip findutils sed libxcb-cursor0

# Unzip and clean up
cd ..

# Append models.py content
cat ~/PRoTECT/ex/benchmarks-deterministic/FOSSIL-versions/models.py >> ~/fossil-main/experiments/benchmarks/models.py

# Install PRoTECT dependencies
cd PRoTECT
pip install --no-index --find-links="dependencies" -r requirements.txt

# Install prerequisites for dreal4
curl -fsSL 'https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install_prereqs.sh' | sudo bash

# Install fossil-main
cd ../fossil-main
pip3 install .

cd ~

echo 'export PYTHONPATH=$PYTHONPATH:$HOME/PRoTECT:$HOME/fossil-main' >> .profile
echo 'export PYTHONPATH=$PYTHONPATH:$HOME/PRoTECT:$HOME/fossil-main' >> .bashrc

mkdir mosek
