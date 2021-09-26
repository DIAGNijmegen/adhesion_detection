# Parent Image
FROM nvcr.io/nvidia/pytorch:20.11-py3

# Install nnU-net
ENV OMP_NUM_THREADS 1
RUN pip3 install --no-cache-dir nnunet==1.6.6

# Download pretrained model
ENV RESULTS_FOLDER /home/user/nnunet/model/results

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
