FROM nvcr.io/nvidia/pytorch:19.09-py3

ENV PATH $PATH:/root/.local/bin
RUN apt-get update && apt-get install -y vim

RUN pip install --user \
    comet-ml \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm

COPY . /playground/
WORKDIR /playground/
