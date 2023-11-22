# eo4all-potential-et
 Potential ET processor for ADB's EO4ALL project

## Building Docker image
     docker build . -t eo4all-potential-et

## Running Docker image
    docker run --network=adb_geonode_network -v /home/processing_hd1:/data eo4all-potential-et

## Develop:
* Build the docker image
* Download input files

### Run just the process without kafka:
```bash
docker run --rm --entrypoint python -v ./data:/data eo4all-potential-et /app/potential_et.py '{"aoi_name": "T50MRV",  "date": "2022-09-11"}'
```

### Run the process with kafka:
```bash
docker run --rm --network=adb_geonode_network -v ./data:/data eo4all-potential-et
```
