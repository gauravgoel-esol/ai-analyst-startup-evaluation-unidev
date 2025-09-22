import logging
import os
from datetime import datetime


"""
This script is used for logging.

"""
# creating log file name
LOG_FILE_NAME = f"{datetime.now().strftime('%d%m%Y__%H%M%S')}.log"
# creating directory
LOG_FILE_DIR=os.path.join(os.getcwd(),'logs')
# Checking if directory exists, then creating if not exists
os.makedirs(LOG_FILE_DIR,exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_FILE_DIR,LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s %(message)s",
    level=logging.INFO,
)