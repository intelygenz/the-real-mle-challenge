import logging

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)
terminal_handler = logging.StreamHandler()
terminal_handler.name = "terminal_handler"
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
terminal_handler.setFormatter(formatter)
logger.addHandler(terminal_handler)
