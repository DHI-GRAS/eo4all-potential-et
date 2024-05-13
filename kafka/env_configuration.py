import os


class Config(object):
    NAME = 'default'
    KAFKA_URI = os.getenv("KAFKA_URI", "kafka1:29092,kafka2:29093")
    TOPIC_ORDER = "PR13_PotentialEvapotranspiration_order_topic"
    TOPIC_SERVICE = "processor_service_topic"
    CONSUMER_GROUP = "geonode" if os.getenv("CONSUMER_GROUP") is None else os.getenv("CONSUMER_GROUP")

    @staticmethod
    def from_profile(profile: str):
        if profile.upper() == 'DEFAULT':
            return Config()
        elif profile.upper() == 'LOCAL':
            return LocalConfig()
        elif profile.upper() == 'PROD':
            return ProductionConfig()
        else:
            raise ValueError("Unexpected environment: " + profile)


class LocalConfig(Config):
    NAME = 'local'
    KAFKA_URI = "localhost:9092"
    TOPIC_TEST_NAME = "topic_test"
    CONSUMER_GROUP = "geonode"


class ProductionConfig(Config):
    NAME = 'prod'
    KAFKA_URI = os.getenv("KAFKA_URI", "kafka1:29092,kafka2:29093")
    TOPIC_TEST_NAME = "topic_test"
    CONSUMER_GROUP = "geonode"


cur_profile = os.environ.get("PROFILE", "default")
SELECTED_CONFIG = Config.from_profile(cur_profile)
