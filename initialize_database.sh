# We install postgres and its dev tools
sudo apt-get -y -qq update
sudo apt-get -y -qq install postgresql postgresql-server-dev-all

#  Start postgres
sudo service postgresql start

# Create user, password, and db
sudo -u postgres psql -U postgres -c "ALTER USER postgres PASSWORD 'postgres';"
sudo -u postgres psql -U postgres -c 'DROP DATABASE IF EXISTS structdb;'
sudo -u postgres psql -U postgres -c 'CREATE DATABASE structdb;'

git clone --recursive https://github.com/lanterndata/lantern.git
cd lantern
mkdir build
cd build
pwd
cmake ..
sudo make install

pip install sentence-transformers==2.2.2