# eo4all-potential-et
 Potential ET processor for ADB's EO4ALL project

## Building Docker image
     docker build . -t eo4all-potential-et

## Running Docker image
    docker run --rm -it eo4all-potential-et '{"aoi_name": "T49MFN", "date":"2022-02-01", "ftp_url":"URL", "ftp_port": "PORT", "ftp_username": "USERNAME", "ftp_pass": "PASSWORD"}'
