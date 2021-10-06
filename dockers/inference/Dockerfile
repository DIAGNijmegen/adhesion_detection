FROM doduo1.umcn.nl/nnunet/sol:latest

ENV CINE_MRI_ARCHIVE=/mnt/netcache/pelvis/projects/bram/archives/cinemri_mha

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY src /home/user/

RUN mkdir -p /home/user/data

COPY image.mha /home/user/data

RUN mkdir -p /home/user/nnunet

RUN mkdir -p /home/user/nnunet/model

COPY nnunet/results /home/user/nnunet/model


# Add cinemri and cinemri-registration libraries to python path
RUN mkdir -p /home/user/.local/lib/python3.8/site-packages/ && \
    printf "/mnt/netcache/pelvis/projects/evgenia/repos/abdomenmrus-cinemri/\n/mnt/netcache/pelvis/projects/evgenia/repos/abdomenmrus-cinemri-registration/\n" > /home/user/.local/lib/python3.8/site-packages/abdomenmrus-cinemri.pth && \
    chown -R user:user /home/user

# Configure inference
RUN chmod +x /home/user/inference.py && \
    ln -s /home/user/inference.py /usr/local/bin/inference
