# eo4all-potential-et
 Potential ET processor for ADB's EO4ALL project

## Building Docker image
     docker build . -t eo4all-potential-et

## Running Docker image
    docker run --network=adb_geonode_network -v /home/processing_hd1:/data eo4all-potential-et
