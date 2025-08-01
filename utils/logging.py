import logging


class ColoredFormatter(logging.Formatter):
    """A custom logging formatter that adds colors based on log level."""

    # ANSI escape codes for colors
    BLUE = "\033[0;34m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    RESET = "\033[0m"

    # The format string for the log message
    log_format_prefix = "[%(levelname)s:%(name)s:%(funcName)s:L.%(lineno)d] "

    # A dictionary to map log levels to colors
    PREFIX_COLORS = {
        logging.INFO: BLUE,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def format(self, record):
        # Get the color for the current log level's prefix
        color = self.PREFIX_COLORS.get(record.levelno, "")
        prefix_formatter = logging.Formatter(self.log_format_prefix)
        prefix = prefix_formatter.format(record)
        colored_prefix = f"{color}{prefix}{self.RESET}"

        # The message itself might have its own colors, which will be preserved.
        message = record.getMessage().lstrip('\n')
        return f"{colored_prefix}{message}"