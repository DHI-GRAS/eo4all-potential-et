FROM python:3.11-slim-bullseye

WORKDIR /app

COPY ./test_send_message/requirements.txt /app/requirements.txt

RUN pip install pip -U \
    && pip install -r requirements.txt

COPY ./kafka /app/kafka
COPY ./test_send_message/*.py /app/

CMD ["python", "sendMsgPlatformDemo.py"]
