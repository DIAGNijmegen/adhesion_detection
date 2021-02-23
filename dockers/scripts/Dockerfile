FROM doduo1.umcn.nl/uokbaseimage/base:latest
#FROM python:3.8-slim-buster

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . /home/user

#WORKDIR /mnt/netcache/pelvis/projects/evgenia/abdomenmrus-cavity-segmentation

# CMD [ "python3", "src/hello.py"]
