import sys 
import logging
import os
from logger import logger

# Ensure the logs folder exists
os.makedirs("logs", exist_ok=True)

# Configure logging BEFORE using it
logging.basicConfig(
    filename="logs/error.log",
    filemode="a",  # append mode
    level=logging.ERROR,  # capture error and above
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def error_message_detail(error, error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename 
    error_message = "Error in [{0}] line [{1}] Message[{2}]".format(
        file_name, exc_tb.tb_lineno,str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message

if __name__=="__main__":
    try:
        a=1/0
    except Exception as e:
        logger.error("Exception occurred: %s", e)
        raise CustomException(e,sys)
        
    