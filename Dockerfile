FROM python:3.10

WORKDIR /tmp
RUN git clone https://github.com/Radiomics/pyradiomics
WORKDIR /tmp/pyradiomics
RUN pip install -r requirements.txt
RUN python setup.py install

RUN pip install arkitekt==0.3.7
ENV fakts_endpoint = "http://herre:8000/f/"