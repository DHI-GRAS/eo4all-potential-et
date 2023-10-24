FROM condaforge/mambaforge:23.3.1-1

RUN mamba install -c conda-forge gdal
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
