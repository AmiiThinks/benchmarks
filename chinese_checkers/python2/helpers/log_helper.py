# log helper
import os
import logging

def setup_logger(name, path, file_name, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(os.path.join(path, file_name))     
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def get_logger(name):
    """To get a logger by name"""

    return logging.getLogger(name)

def log(name, message):
    """To log a message"""

    logger = get_logger(name)
    logger.info(message)

