FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git gcc make python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Build libsmctrl (discovery functions only — no masking used here)
RUN git clone http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git /opt/libsmctrl \
    && cd /opt/libsmctrl && make libsmctrl.so

COPY . /opt/maskmaster

# Rebuild libmaskmaster.so for this platform with discovery enabled
RUN cd /opt/maskmaster/c \
    && make clean \
    && make DISCOVERY=1 SMCTRL_DIR=/opt/libsmctrl

RUN pip3 install --no-deps -e /opt/maskmaster/python

ENV LD_LIBRARY_PATH=/opt/libsmctrl:/opt/maskmaster/c

EXPOSE 8000

# nvdebug must be loaded on the host before running this container.
# Run with:
#   docker run --gpus all -v /proc/gpu0:/proc/gpu0 -p 8000:8000 maskmaster
CMD ["python3", "-m", "maskmaster.gui", "--host", "0.0.0.0"]
