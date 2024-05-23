cd ..
sudo apt-get update
sudo apt-get install -y python3 python3-pip findutils sed libxcb-cursor0
pip install --no-index --find-links="dependencies" -r requirements.txt

cd ~

echo 'export PYTHONPATH=$PYTHONPATH:$HOME/PRoTECT:$HOME/fossil-main' >> .profile
echo 'export PYTHONPATH=$PYTHONPATH:$HOME/PRoTECT:$HOME/fossil-main' >> .bashrc

mkdir mosek
