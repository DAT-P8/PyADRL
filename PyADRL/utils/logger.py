import logging


class Logger:
    logger: logging.Logger = logging.getLogger("ray")

    @staticmethod
    def debug(msg: str):
        Logger.logger.debug("Debug: " + msg)

    @staticmethod
    def info(msg: str):
        Logger.logger.info("Info: " + msg)

    @staticmethod
    def warning(msg: str):
        Logger.logger.warning("Warning: " + msg)

    @staticmethod
    def error(msg: str):
        Logger.logger.error("Error: " + msg)

    @staticmethod
    def critical(msg: str):
        Logger.logger.critical("Critical: " + msg)