import logging


class ColoredFormatter(logging.Formatter):
    """A custom logging formatter that adds colors based on log level."""

    # ANSI escape codes for colors
    BLUE = "\033[0;34m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    RESET = "\033[0m"

    # The format string for the log message
    log_format = "[%(levelname)s:%(name)s:%(funcName)s:Line %(lineno)d] %(message)s"

    # A dictionary to map log levels to colors
    FORMATS = {
        logging.INFO: BLUE + log_format + RESET,
        logging.WARNING: YELLOW + log_format + RESET,
        logging.ERROR: RED + log_format + RESET,
        logging.CRITICAL: RED + log_format + RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.log_format)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)