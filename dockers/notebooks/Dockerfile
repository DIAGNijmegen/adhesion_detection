FROM doduo1.umcn.nl/uokbaseimage/base:latest

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . /home/user/

# WORKDIR /mnt/netcache/pelvis/projects/evgenia/abdomenmrus-cavity-segmentation

# ENTRYPOINT /bin/bash dockers/notebooks/script.sh