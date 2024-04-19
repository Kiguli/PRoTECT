import logging
from colorama import Fore, Style


def setup_logging(logging_mode):
    # Set the logging level
    logging.basicConfig(level=logging_mode)

    # Create a custom formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')

    # Apply the formatter to the root logger
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)

    # Create a color formatter
    class ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: Fore.BLUE,
            logging.INFO: Fore.GREEN,
            logging.WARNING: Fore.YELLOW,
            logging.ERROR: Fore.RED,
            logging.CRITICAL: Fore.RED,
        }

        def format(self, record):
            log_color = self.COLORS.get(record.levelno, '')
            log_reset = Style.RESET_ALL
            record.msg = log_color + record.msg + log_reset
            return super().format(record)

    # Apply the color formatter to the root logger
    color_formatter = ColorFormatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    for handler in logging.root.handlers:
        handler.setFormatter(color_formatter)
