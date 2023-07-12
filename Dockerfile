FROM continuumio/miniconda3:latest

RUN conda config --add channels conda-forge
RUN conda config --set channel_priority strict
RUN conda install -c conda-forge gdal
ENV PROJ_LIB="/opt/conda/share/proj"

RUN apt-get update -y && \
    apt-get install vim gcc -y && \
    apt-get upgrade -y

WORKDIR /app

COPY requirements.txt .

RUN pip install -U pip && pip install -r requirements.txt

COPY *.py /app/
COPY *.csv /app/

ENTRYPOINT ["python", "-u", "potential_et.py"]
