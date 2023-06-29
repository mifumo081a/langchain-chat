touch .env
echo "DIR=$(basename `pwd`)" > .env
echo "UID=$(id -u)" >> .env
echo "GID=$(id -g)" >> .env
echo "UNAME=$(id -un)" >> .env
echo "GNAME=$(id -gn)" >> .env