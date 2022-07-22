FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -n python-app && \
    conda activate python-app && \
    conda install python=3.6 pip && \
    pip install torch==1.10.0+cpu torchvision==0.11.0+cpu torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install opencv-python==4.1.2.30 \
    && pip install onnx \
    && pip install onnxruntime
# VOLUME ./ /home:Z
# WORKDIR /home
# CMD ["python", "test.py"]
ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]