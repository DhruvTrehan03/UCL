import sys
import time

import numpy as np
import pyeit.eit.bp as bp
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
import serial
from colorama import Fore, Style, just_fix_windows_console

just_fix_windows_console()


def print_info(msg: str) -> None:
    print(f"{Fore.CYAN}INFO: {msg}{Style.RESET_ALL}")


def print_warning(msg: str) -> None:
    print(f"{Fore.YELLOW}WARN: {msg}{Style.RESET_ALL}")


def print_error(msg: str) -> None:
    print(f"{Fore.RED}ERROR: {msg}{Style.RESET_ALL}")


class EITDevice:
    def __init__(self, port: str):
        self.port = port
        self.version = "N/A"
        self.build_date_time = "N/A"
        self._ser = None
        self.latest_frame = None

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def start_streaming(self, data_callback):
        pass


frame_count = 0
