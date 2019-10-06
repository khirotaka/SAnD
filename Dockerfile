FROM nvcr.io/nvidia/pytorch:19.09-py3

ENV PATH $PATH:/root/.local/bin
RUN apt-get update && \
    apt-get install -y --no-install-recommends vim=8.0.1365 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --user comet-ml==2.0.14 \
                       pandas==0.25.1 \
                       scikit-learn==0.21.3 \
                       matplotlib==3.1.1 \
                       seaborn==0.9.0 \
                       tqdm==4.36.1

COPY . /playground/
WORKDIR /playground/
