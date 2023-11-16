class MessageValueException(Exception):
    def __str__(self):
        return "It seems you used a non JSON serializable object"
