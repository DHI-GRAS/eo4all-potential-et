# eo4all-potential-et
 Potential ET processor for ADB's EO4ALL project

## Building Docker image
     docker build . -t docker_pr13_potential-evapotranspiration_v1

## Running Docker image
    docker run --network=adb_geonode_network -v /home/processing_hd1:/data docker_pr13_potential-evapotranspiration_v1

## Develop:
* Build the docker image
* Download input files

### Run just the process without kafka:
```bash
docker run --rm --entrypoint python -v ./data:/data docker_pr13_potential-evapotranspiration_v1 /app/potential_et.py '{"aoi_name": "T50MRV",  "date": "2022-09-11"}'
```

### Run the process with kafka:
```bash
docker run --rm --network=adb_geonode_network -v ./data:/data docker_pr13_potential-evapotranspiration_v1
```


## Useful
Test the process:
```bash
git pull && docker build . -t docker_pr13_potential-evapotranspiration_test_prcess && docker run --rm --network=adb_geonode_network -e DEBUG=True -e CONSUMER_GROUP=et_test_group_1 -v /home/processing_hd1:/data docker_pr13_potential-evapotranspiration_test_prcess
```

Send testing message and check the logs:
```bash
make run && sleep 10 && tail -f $(ls -rt /home/processing_hd1/logs/docker_PR13_Potential-Evapotranspiration_logs/Potential_Evapotranspiration_*  | tail -n 1)
```
