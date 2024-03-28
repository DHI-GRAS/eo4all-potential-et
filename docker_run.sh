docker run \
    -d \
    --rm \
    --name docker_pr13_potential-evapotranspiration_v1 \
    --network=adb_geonode_network \
    -v /home/processing_hd2/_ancillaries:/data/_ancillaries \
    -v /home/processing_hd2/logs:/data/logs \
    -v /home/processing_hd2/testFolder:/data/testFolder \
    -v /home/processing_hd3/outputs:/data/outputs \
    docker_pr13_potential-evapotranspiration_v1
