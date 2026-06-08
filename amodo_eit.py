import queue
import threading
import time
from typing import Callable
import serial
from serial.tools import list_ports
import numpy as np
from utils import EITDevice, print_info, print_warning

EIT_USB_VID = 0x0483  # EIT USB VID: same as STM32's VID
EIT_USB_PID = 0x5740  # EIT USB PID: same as STM32 HS VCP
TIMEOUT = 0.1  # seconds for serial read timeout


def _find_candidate_ports() -> list[str]:
    """Return a list of port names whose USB VID/PID match."""
    candidate_ports = []

    for p in list_ports.comports():
        # Some ports may not have VID/PID info
        if p.vid is None or p.pid is None:
            continue

        if p.vid == EIT_USB_VID and p.pid == EIT_USB_PID:
            candidate_ports.append(p.device)

    return candidate_ports


class AmodoEITDevice(EITDevice):
    def __init__(self, port: str):
        self.port = port
        self.version = ""
        self.build_date_time = ""
        self._ser = None

        self._streaming = False
        self._stream_thread = None
        self._data_queue = queue.Queue()
        self.latest_frame = None

    def __enter__(self):
        """Open the serial port when entering the context."""
        self._ser = serial.Serial(self.port, 115200, timeout=TIMEOUT)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the serial port when exiting the context."""
        if self._ser and self._ser.is_open:
            # self._run_command("stop", False)
            self.stop_streaming()
            self._ser.close()
        return False  # Don't suppress exceptions

    def _ensure_open(self) -> None:
        """Ensure the serial port is open."""
        if self._ser is None or not self._ser.is_open:
            raise RuntimeError(f"Serial port {self.port} not open. Use 'with AmodoEITDevice(...)' context manager.")

    def _run_command(self, command: str, waitForResponse: bool = True) -> str:
        """
        Send a command to the device and return the response.
        Must be called within a context manager (with statement).
        """
        self._ensure_open()

        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

        self._ser.write((command + "\r").encode())
        self._ser.flush()

        if not waitForResponse:
            return ""

        start_time = time.perf_counter()
        while True:
            line = self._ser.readline()
            if line.startswith(b"OK:"):
                break
            if line.startswith(b"ERROR:"):
                raise RuntimeError(f"{self.port}: command '{command}' failed -> {line.decode(errors='replace').strip()}")
            if time.perf_counter() - start_time > TIMEOUT:
                raise TimeoutError(f"{self.port}: timeout waiting for response to command '{command}'")

        return line.decode(errors="replace").strip()

    def reset(self) -> None:
        """Ensures the device is in a known state, regardless of whether it was previously streaming or loading configurations."""
        self._ensure_open()

        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

        self._run_command("", False)  # Send blank line in case we're in `load` mode
        self._run_command("stop", False)  # Send "stop" in case we're in streaming mode
        self._run_command("reset", False)

        start_time = time.perf_counter()
        while True:
            line = self._ser.readline()
            if line.startswith(b"OK: Reset"):
                break
            if time.perf_counter() - start_time > TIMEOUT:
                raise TimeoutError(f"{self.port}: timeout waiting for response to 'reset' command")

    def set_electrode_configurations(self, configurations: list[tuple[int, int, int, int, str, str]]) -> None:
        """Send electrode configurations to the device."""
        self._ensure_open()

        self._ser.write("load\r".encode())
        for A, B, M, N, TX_GAIN, RX_GAIN in configurations:
            self._ser.write(f"{A} {B} {M} {N} {TX_GAIN} {RX_GAIN}\r".encode("ascii"))
        self._ser.write(b"\r")  # Terminate load command
        self._ser.flush()
        response = self._ser.read_until(b"\n").decode("ascii").strip()
        if not response.startswith("OK:"):
            raise RuntimeError(f"{self.port}: unexpected response -> {response}")

    def set_stimulation_frequency(self, freq_khz: int) -> None:
        """Set the stimulation frequency in kHz."""
        self._run_command(f"freqkhz {freq_khz}")

    def set_num_periods_to_sample_per_measurement(self, periods: int) -> None:
        """Set the number of periods to sample per measurement."""
        self._run_command(f"setperiods {periods}")

    def do_autogain(self) -> None:
        """Perform auto-gain calibration."""
        self._run_command("autogain", False)
        self._ser.write(b"\r")  # Send blank line to start autogain
        print(self._ser.readline().decode("ascii").strip())
        while True:
            line = self._ser.readline()
            if line.startswith(b"OK: Autogain complete."):
                break
        print_info(f"{self.port}: autogain complete.")

    def start_streaming(self, callback: Callable[[bytes], None] | None = None) -> None:
        """
        Start reading data continuously in a background thread.

        Args:
            callback: Optional function to call with each line of data.
                     If None, data goes into self._data_queue
        """
        self._ensure_open()

        if self._streaming:
            raise RuntimeError(f"{self.port}: already streaming")

        self._run_command("start", False)
        self._streaming = True
        self._stream_thread = threading.Thread(
            target=self._stream_worker,
            args=(callback,),
            daemon=True,  # Thread will exit when main program exits
        )
        self._stream_thread.start()

    def stop_streaming(self) -> None:
        """Stop the streaming thread."""
        if not self._streaming:
            return

        self._streaming = False

        if self._stream_thread:
            self._stream_thread.join()
            if self._stream_thread.is_alive():
                print_warning(f"{self.port}: warning - streaming thread didn't stop cleanly")
            self._stream_thread = None

    def _stream_worker(self, callback: Callable[[bytes], None] | None) -> None:
        """Worker function that runs in the background thread."""
        buffer = bytearray()
        self._ser.timeout = 0.1  # Non-blocking read with short timeout
        while self._streaming:
            try:
                if self._ser.in_waiting > 0:
                    chunk = self._ser.read(self._ser.in_waiting)
                    buffer.extend(chunk)
                else:
                    # Nothing available, do a blocking read with short timeout
                    chunk = self._ser.read(1)
                    if chunk:
                        buffer.extend(chunk)
                    else:
                        continue  # Timeout, check _streaming flag

                while b"\n" in buffer:
                    line_end = buffer.index(b"\n")
                    line_bytes = bytes(buffer[:line_end])
                    del buffer[: line_end + 1]  # Remove processed line from buffer
                    line_str = line_bytes.decode("ascii").strip()
                    if not line_str:
                        continue
                    response_data = line_str.strip().split(",")
                    clipping = any("C" in x for x in response_data)
                    data = np.array([float(x.rstrip("C")) for x in response_data], dtype=float)

                    if callback:
                        callback(data)

                    self.latest_frame = (data, clipping)
            except Exception as e:
                print(f"{self.port}: streaming error -> {e}")
                # self._streaming = False
        print(f"{self.port}: streaming stopped.")


def check_device(port_name) -> tuple[bool, AmodoEITDevice | None]:
    """
    Open port, send commands, read response.
    Returns (ok: bool)
    """
    try:
        with AmodoEITDevice(port_name) as device:
            device.reset()
            version_response = device._run_command("version")
            version_response_expected_prefix = "OK: Amodo EIT "
            if version_response.startswith(version_response_expected_prefix):
                response = version_response[len(version_response_expected_prefix) :].strip()
                version = response.split(" ")[0]
                build_date_time = response[len(version) :].strip().replace("(", "").replace(")", "")
                device.version = version
                device.build_date_time = build_date_time
                return True, device
            else:
                print(f"{port_name}: unexpected response -> {version_response!r}")
                return False, None

    except serial.SerialException as e:
        print(f"{port_name}: serial error -> {e}")
        return False, None


def get_connected_devices() -> list[AmodoEITDevice]:
    """Return a list of ports that are connected to EIT devices."""
    ports = _find_candidate_ports()
    valid_devices: list[AmodoEITDevice] = []
    for port in ports:
        ok, device = check_device(port)
        if ok:
            valid_devices.append(device)
    return valid_devices
