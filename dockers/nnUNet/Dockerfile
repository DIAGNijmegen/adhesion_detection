FROM doduo1.umcn.nl/nnunet/sol:latest

ENV CINE_MRI_ARCHIVE=/mnt/netcache/pelvis/projects/bram/archives/cinemri_mha

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY src /home/user/

# Add cinemri library to python path
RUN mkdir -p /home/user/.local/lib/python3.8/site-packages/ && \
    printf "/mnt/netcache/pelvis/projects/evgenia/repos/abdomenmrus-cinemri/\n/mnt/netcache/pelvis/projects/evgenia/repos/abdomenmrus-cinemri-registration/\n" > /home/user/.local/lib/python3.8/site-packages/abdomenmrus-cinemri.pth && \
    chown -R user:user /home/user

# Configure data conversion
RUN chmod +x /home/user/data_conversion.py && \
    ln -s /home/user/data_conversion.py /usr/local/bin/data_conversion

# Configure visceral slide functions
RUN chmod +x /home/user/visceral_slide_pipeline.py && \
    ln -s /home/user/visceral_slide_pipeline.py /usr/local/bin/visceral_slide

# Configure segmentation functions
RUN chmod +x /home/user/segmentation.py && \
    ln -s /home/user/segmentation.py /usr/local/bin/segmentation

# Configure utils functions
RUN chmod +x /home/user/utils.py && \
    ln -s /home/user/utils.py /usr/local/bin/utils
