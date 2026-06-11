import argparse
import csv
import os
import queue
import re
import threading
import time
from datetime import datetime

import numpy as np
import serial
import pyeit.eit.protocol as protocol
import amodo_eit as eit
from utils import print_info
import random


# ---------------------------------------------------------------------------
# User-configurable experiment settings
# ---------------------------------------------------------------------------
FILE_NAME = "testing"

TARGET_FORCES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
MEASUREMENTS_PER_FORCE = 300

CORNER_X = 45
CORNER_Y = 9
START_Z = 10
RETRACT_HEIGHT = 5
FORCE_STEP_MM = 0.05
MAX_FORCE_TRAVEL_MM = 2.0
SETTLE_TIME_S = 1.0
EIT_AVERAGE_FRAMES = 3

NUM_RANDOM_POINTS = (len(TARGET_FORCES)* MEASUREMENTS_PER_FORCE)
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# EIT device settings
# ---------------------------------------------------------------------------
NUM_ELECTRODES = 16
INJ_STEP = 1
READ_STEP = 1
STIM_FREQ_KHZ = 50
PERIODS_PER_MEASUREMENT = 50
TX_GAIN = 32
RX_GAIN = 2




#----------------------------------------------------------------------------
# Random point and force generator
#----------------------------------------------------------------------------
def triangle_area(p1, p2, p3):
    return abs(
        (p2[0] - p1[0]) * (p3[1] - p1[1])
        - (p3[0] - p1[0]) * (p2[1] - p1[1])
    ) / 2.0


def random_point_in_triangle(p1, p2, p3):
    """
    Uniform random point inside a triangle.
    """
    r1 = random.random()
    r2 = random.random()

    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2

    x = (
        p1[0]
        + r1 * (p2[0] - p1[0])
        + r2 * (p3[0] - p1[0])
    )

    y = (
        p1[1]
        + r1 * (p2[1] - p1[1])
        + r2 * (p3[1] - p1[1])
    )

    return x, y


def generate_random_points_in_quadrilateral(n_points, seed=42):
    """
    Generate reproducible uniformly-distributed points
    inside the quadrilateral:

        (0,0)
        (0,20)
        (70,17.5)
        (70,2.5)
    """

    random.seed(seed)

    A = (0.0, 0.0)
    B = (0.0, 20.0)
    C = (70.0, 17.5)
    D = (70.0, 2.5)

    # Split into triangles ABC and ACD
    area1 = triangle_area(A, B, C)
    area2 = triangle_area(A, C, D)

    points = []

    for _ in range(n_points):

        if random.random() < area1 / (area1 + area2):
            p = random_point_in_triangle(A, B, C)
        else:
            p = random_point_in_triangle(A, C, D)

        points.append(p)

    return points

def generate_force_schedule():
    random.seed(RANDOM_SEED)

    schedule = []

    for force in TARGET_FORCES:
        schedule.extend(
            [force] * MEASUREMENTS_PER_FORCE
        )

    random.shuffle(schedule)

    return schedule

RANDOM_POINTS = generate_random_points_in_quadrilateral(
    NUM_RANDOM_POINTS,
    RANDOM_SEED,
)

FORCE_SCHEDULE = generate_force_schedule()

MEASUREMENT_SCHEDULE = list(
    zip(
        RANDOM_POINTS,
        FORCE_SCHEDULE,
    )
)

# ---------------------------------------------------------------------------
# EIT device thread
# ---------------------------------------------------------------------------

class AmodoEITDevice(threading.Thread):
    def __init__(self, q_out, group=None):
        super().__init__(group=group, name="AmodoEITDevice", daemon=True)
        self.devices = eit.get_connected_devices()
        self.q = q_out
        self.stop_evt = threading.Event()
        if not self.devices:
            print_info("No Amodo EIT devices connected.")
            raise SystemExit(1)
        if len(self.devices) > 1:
            print_info("Multiple Amodo EIT devices detected.")
        self.device = self.devices[0]

        self.protocol_obj = protocol.create(
            NUM_ELECTRODES,
            dist_exc=INJ_STEP,
            step_meas=READ_STEP,
            parser_meas="rotate_meas",
        )
        self.baseline_frame = None
        self.baseline_clipping = None

    def _configure_and_start_streaming(self):
        print_info(
            f"Using device: {self.device.port}, "
            f"version {self.device.version}, build {self.device.build_date_time}"
        )
        self.device.set_stimulation_frequency(STIM_FREQ_KHZ)

        print_info("Loading electrode configuration...")
        electrode_configurations = []
        for i_exc, exc_pair in enumerate(self.protocol_obj.ex_mat):
            A, B = exc_pair
            meas_pairs = self.protocol_obj.meas_mat[i_exc]
            for M, N in meas_pairs:
                pin_offset = 1
                configuration = (
                    A + pin_offset, B + pin_offset,
                    M + pin_offset, N + pin_offset,
                    TX_GAIN, RX_GAIN,
                )
                electrode_configurations.append(configuration)

        self.device.set_electrode_configurations(electrode_configurations)
        print_info(
            f"Electrode configuration loaded ({len(electrode_configurations)} configurations).\n"
        )
        self.device.set_num_periods_to_sample_per_measurement(PERIODS_PER_MEASUREMENT)

        print_info("Capturing baseline frame...")
        self.device.start_streaming()

        while self.device.latest_frame is None and not self.stop_evt.is_set():
            time.sleep(0.01)

        if self.device.latest_frame is not None:
            baseline_frame, baseline_clipping = self.device.latest_frame
            baseline_frame = np.array(
                [x if x > 1e-12 else 1e-12 for x in baseline_frame]
            )
            if baseline_clipping:
                print_info("Clipping detected in baseline")
            print_info(f"Baseline captured: {len(baseline_frame)} measurements\n")
            self.baseline_frame = baseline_frame
            self.baseline_clipping = baseline_clipping
            baseline_sample = {
                "t": time.monotonic(),
                "readings": baseline_frame.tolist(),
                "clipping": baseline_clipping,
                "baseline": True,
            }
            try:
                self.q.put_nowait(baseline_sample)
            except queue.Full:
                pass

    def run(self):
        last_frame = None
        try:
            with self.device:
                self._configure_and_start_streaming()
                while not self.stop_evt.is_set():
                    latest = self.device.latest_frame
                    if latest is None:
                        time.sleep(0.002)
                        continue
                    frame, clipping = latest
                    if frame is None:
                        time.sleep(0.002)
                        continue
                    frame_arr = np.asarray(frame, dtype=float)
                    if last_frame is not None and np.array_equal(frame_arr, last_frame):
                        time.sleep(0.002)
                        continue
                    last_frame = frame_arr.copy()
                    sample = {
                        "t": time.monotonic(),
                        "readings": frame_arr.tolist(),
                        "clipping": clipping,
                    }
                    try:
                        self.q.put_nowait(sample)
                    except queue.Full:
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            pass
                        self.q.put_nowait(sample)
                    time.sleep(0.001)
        except KeyboardInterrupt:
            print_info("\n\nStopped by user")
        except Exception as e:
            print_info(f"Unexpected error in EIT reader thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                self.device.stop_streaming()
            except Exception:
                pass

    def stop(self):
        self.stop_evt.set()
        try:
            self.device.stop_streaming()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Force reader thread
# ---------------------------------------------------------------------------

class ForceReader(threading.Thread):
    def __init__(self, force_serial, q_out, group=None):
        super().__init__(group=group, name="ForceReader", daemon=True)
        self.force_serial = force_serial
        self.q = q_out
        self.stop_evt = threading.Event()

    def run(self):
        self.force_serial.flushInput()
        while not self.stop_evt.is_set():
            try:
                line = self.force_serial.readline()
                if not line:
                    time.sleep(0.01)
                    continue
                try:
                    force_value = float(line.strip())
                except ValueError:
                    continue
                sample = {
                    "t": time.monotonic(),
                    "force": force_value,
                }
                try:
                    self.q.put_nowait(sample)
                except queue.Full:
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                    self.q.put_nowait(sample)
            except Exception as exc:
                print_info(f"Unexpected error in force reader thread: {exc}")
                time.sleep(0.05)

    def stop(self):
        self.stop_evt.set()


# ---------------------------------------------------------------------------
# Printer controller
# ---------------------------------------------------------------------------

class PrinterController:
    def __init__(self, printer_serial):
        self.printer_serial = printer_serial

    def write(self, command):
        """Send a raw bytes command to the printer."""
        self.printer_serial.write(command)

    def wait_for_position(self):
        """Block until the printer confirms its current position."""
        self.printer_serial.flush()
        self.write(b"M114 R\r\n")
        while True:
            line = self.printer_serial.readline()
            if not line:
                continue
            decoded = line.decode(errors="ignore").strip()
            if "Count" in decoded:
                break

    def get_position(self):
        """Send M114 R and parse the returned X, Y, Z coordinates.

        Returns
        -------
        tuple[float, float, float]
            (actual_x, actual_y, actual_z) in mm.
        """
        self.printer_serial.flush()
        self.write(b"M114 R\r\n")
        while True:
            line = self.printer_serial.readline()
            if not line:
                continue
            decoded = line.decode(errors="ignore").strip()
            # Marlin response looks like:
            # X:70.00 Y:28.00 Z:20.00 E:0.00 Count X:5600 Y:2240 Z:8000
            match = re.search(
                r"X:([-\d.]+)\s+Y:([-\d.]+)\s+Z:([-\d.]+)", decoded
            )
            if match:
                actual_x = float(match.group(1))
                actual_y = float(match.group(2))
                actual_z = float(match.group(3))
                return actual_x, actual_y, actual_z

    def close(self):
        try:
            self.printer_serial.close()
        except Exception:
            pass

#----------------------------------------------------------------------------
# Random point generator
#----------------------------------------------------------------------------
def triangle_area(p1, p2, p3):
    return abs(
        (p2[0] - p1[0]) * (p3[1] - p1[1])
        - (p3[0] - p1[0]) * (p2[1] - p1[1])
    ) / 2.0


def random_point_in_triangle(p1, p2, p3):
    """
    Uniform random point inside a triangle.
    """
    r1 = random.random()
    r2 = random.random()

    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2

    x = (
        p1[0]
        + r1 * (p2[0] - p1[0])
        + r2 * (p3[0] - p1[0])
    )

    y = (
        p1[1]
        + r1 * (p2[1] - p1[1])
        + r2 * (p3[1] - p1[1])
    )

    return x, y


def generate_random_points_in_quadrilateral(n_points, seed=42):
    """
    Generate reproducible uniformly-distributed points
    inside the quadrilateral:

        (0,0)
        (0,20)
        (70,17.5)
        (70,2.5)
    """

    random.seed(seed)

    A = (0.0, 0.0)
    B = (0.0, 20.0)
    C = (70.0, 17.5)
    D = (70.0, 2.5)

    # Split into triangles ABC and ACD
    area1 = triangle_area(A, B, C)
    area2 = triangle_area(A, C, D)

    points = []

    for _ in range(n_points):

        if random.random() < area1 / (area1 + area2):
            p = random_point_in_triangle(A, B, C)
        else:
            p = random_point_in_triangle(A, C, D)

        points.append(p)

    return points



# ---------------------------------------------------------------------------
# EIT helper
# ---------------------------------------------------------------------------


def collect_average_eit_sample(q_eit, n_frames):
    """Flush stale frames then collect and average n_frames fresh EIT frames.

    Parameters
    ----------
    q_eit : queue.Queue
        Queue fed by AmodoEITDevice.
    n_frames : int
        Number of fresh frames to average.

    Returns
    -------
    dict
        {
            "t": float,          # monotonic timestamp of the final frame
            "readings": list,    # per-channel averages
            "clipping": bool,    # True if any frame had clipping
        }
    """
    # Flush stale samples
    while True:
        try:
            q_eit.get_nowait()
        except queue.Empty:
            break

    frames = []
    any_clipping = False

    while len(frames) < n_frames:
        sample = q_eit.get()  # blocks until a fresh frame arrives
        frames.append(sample)
        if sample.get("clipping"):
            any_clipping = True

    values = np.mean(
        np.array([f["readings"] for f in frames], dtype=float), axis=0
    )
    return {
        "t": frames[-1]["t"],
        "readings": values.tolist(),
        "clipping": any_clipping,
    }


def sample_force_now(q_force):
    """Drain the force queue and return the most recent sample (or None)."""
    last = None
    while True:
        try:
            last = q_force.get_nowait()
        except queue.Empty:
            break
    return last


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def write_csv_header(csv_path, channel_count):
    header = [
        "target_x_mm", "target_y_mm", "target_force_N",
        "actual_x_mm", "actual_y_mm", "position_time_s",
        "actual_force_N", "force_time_s",
    ]
    header.extend([f"eit_{i}" for i in range(channel_count)])
    header.append("eit_time_s")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)


def append_csv_row(
    csv_path,
    target_x, target_y, target_force,
    actual_x, actual_y, position_time_s,
    actual_force, force_time_s,
    eit_sample, channel_count, experiment_start,
):
    eit_time_s = eit_sample["t"] - experiment_start if eit_sample else ""
    eit_values = eit_sample["readings"] if eit_sample else []

    row = [
        target_x, target_y, target_force,
        actual_x, actual_y, position_time_s,
        actual_force, force_time_s,
    ]
    for i in range(channel_count):
        row.append(f"{eit_values[i]:.12f}" if i < len(eit_values) else "")
    row.append(eit_time_s)

    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ---------------------------------------------------------------------------
# Setup / teardown helpers
# ---------------------------------------------------------------------------

def setup(printer):
    printer.write(
        f"G1 Z{START_Z + RETRACT_HEIGHT} F400\r\n".encode()
    )
    print_info("Homing...")
    printer.write(b"G28\r\n")
    printer.wait_for_position()
    print_info("Homed.")
    printer.write(
        f"G1 Z{START_Z + RETRACT_HEIGHT} F400\r\n".encode()
    )
    printer.write(
        f"G1 X{CORNER_X} Y{CORNER_Y} F1000\r\n".encode()
    )
    printer.wait_for_position()
    print_info("Setup complete.")


def retract_probe(printer):
    printer.write(
        f"G1 Z{START_Z + RETRACT_HEIGHT} F400\r\n".encode()
    )
    printer.wait_for_position()


def remove_probe(printer):
    printer.write(
        f"G1 X{CORNER_X + 19} Y{CORNER_Y} F800\r\n".encode()
    )
    printer.wait_for_position()
    printer.write(
        f"G1 Z{START_Z + 1} F400\r\n".encode()
    )
    printer.wait_for_position()


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def take_measurement(target_force, printer, q_force, q_eit, experiment_start):
    """Lower probe until target_force is reached, then collect averaged EIT.

    Returns
    -------
    dict with keys: actual_force, force_time_s, eit_sample
    """
    # Lower to START_Z
    printer.write(f"G1 Z{START_Z} F400\r\n".encode())
    printer.wait_for_position()

    # Drain stale force readings
    sample_force_now(q_force)

    # Wait for a valid initial force reading
    force_sample = None
    while force_sample is None:
        force_sample = sample_force_now(q_force)
        if force_sample is None:
            time.sleep(0.05)

    force = force_sample["force"]
    travel = 0.0
    current_z = START_Z

    print_info(
        f"  Initial force: {force:.3f} N — probing to {target_force} N..."
    )

    while force < target_force and travel <= MAX_FORCE_TRAVEL_MM:
        current_z -= FORCE_STEP_MM
        travel += FORCE_STEP_MM
        printer.write(f"G1 Z{current_z:.4f} F400\r\n".encode())
        printer.wait_for_position()

        force_sample = None
        while force_sample is None:
            force_sample = sample_force_now(q_force)
            if force_sample is None:
                time.sleep(0.05)

        force = force_sample["force"]
        print_info(f"  Force: {force:.3f} N  (travel {travel:.2f} mm)")

    force_time_s = time.monotonic() - experiment_start
    print_info(
        f"  Target reached: {force:.3f} N at travel {travel:.2f} mm. "
        f"Settling {SETTLE_TIME_S} s..."
    )

    # Stabilisation delay
    time.sleep(SETTLE_TIME_S)

    # Collect and average EIT frames
    eit_sample = collect_average_eit_sample(q_eit, EIT_AVERAGE_FRAMES)

    return {
        "actual_force": force,
        "force_time_s": force_time_s,
        "eit_sample": eit_sample,
    }


# ---------------------------------------------------------------------------
# Measurement runner thread
# ---------------------------------------------------------------------------

class MeasurementRunner(threading.Thread):
    """Run all measurements in a background thread."""

    def __init__(
        self, printer_serial, q_force, q_eit, experiment_start, output_dir
    ):
        super().__init__(daemon=True, name="MeasurementRunner")
        self.printer_serial = printer_serial
        self.q_force = q_force
        self.q_eit = q_eit
        self.experiment_start = experiment_start
        self.output_dir = output_dir
        self.stop_evt = threading.Event()

    def run(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_path = os.path.join(self.output_dir, f"{timestamp}_{FILE_NAME}.csv")

        printer = PrinterController(self.printer_serial)

        # ---- Wait for baseline / determine channel count ----
        channel_count = 0
        baseline_sample = None
        t0 = time.time()
        while time.time() - t0 < 10:
            try:
                samp = self.q_eit.get(timeout=1.0)
                if samp and "readings" in samp and len(samp["readings"]) > 0:
                    channel_count = len(samp["readings"])
                    baseline_sample = samp
                    print_info(f"Detected EIT channel count: {channel_count}")
                    break
            except queue.Empty:
                continue

        if channel_count == 0:
            print_info("No EIT data detected — exiting.")
            return

        # Write header
        write_csv_header(csv_path, channel_count)

        # Write baseline row (position / force fields left empty)
        if baseline_sample is not None:
            append_csv_row(
                csv_path,
                target_x="", target_y="", target_force="",
                actual_x="", actual_y="",
                position_time_s=baseline_sample["t"] - self.experiment_start,
                actual_force="", force_time_s="",
                eit_sample=baseline_sample,
                channel_count=channel_count,
                experiment_start=self.experiment_start,
            )
            print_info(f"Baseline written to {csv_path}")

        try:
            setup(printer)

            # Measurement loop: force → repeat → XY position
            for idx, ((target_x, target_y), target_force) in enumerate(MEASUREMENT_SCHEDULE):

                    if self.stop_evt.is_set():
                        break

                    print_info(
                        f"[{idx+1}/{len(MEASUREMENT_SCHEDULE)}] "
                        f"Target force={target_force:.2f} N "
                        f"Target position=({target_x:.2f}, {target_y:.2f})"
                    )

                    # ---- Move XY ----
                    printer.write(
                        f"G1 X{CORNER_X + target_x} Y{CORNER_Y + target_y} F800\r\n".encode()
                    )
                    printer.wait_for_position()

                    # ---- Read actual position ----
                    actual_x, actual_y, _ = printer.get_position()
                    position_time_s = time.monotonic() - self.experiment_start

                    # ---- Probe and collect EIT ----
                    result = take_measurement(
                        target_force,
                        printer,
                        self.q_force,
                        self.q_eit,
                        self.experiment_start,
                    )

                    # ---- Retract ----
                    retract_probe(printer)

                    # ---- Write CSV row ----
                    append_csv_row(
                        csv_path,
                        target_x=target_x,
                        target_y=target_y,
                        target_force=target_force,
                        actual_x=actual_x,
                        actual_y=actual_y,
                        position_time_s=position_time_s,
                        actual_force=result["actual_force"],
                        force_time_s=result["force_time_s"],
                        eit_sample=result["eit_sample"],
                        channel_count=channel_count,
                        experiment_start=self.experiment_start,
                    )

                    print_info(
                        f"Completed {idx+1}/{len(MEASUREMENT_SCHEDULE)}"
                    )

                    time.sleep(SETTLE_TIME_S)

        except KeyboardInterrupt:
            print_info("Measurement stopped by user.")
        except Exception as e:
            print_info(f"Error in measurement runner: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                remove_probe(printer)
            except Exception:
                pass
            print_info(f"Measurement log saved to {csv_path}")

    def stop(self):
        self.stop_evt.set()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Synchronize EIT samples with printer control"
    )
    parser.add_argument(
        "--printer-port", default="/dev/ttyUSB0",
        help="Serial port for printer controller",
    )
    parser.add_argument(
        "--force-port", default="/dev/ttyACM0",
        help="Serial port for force sensor",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for CSV logs",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(
        script_dir, "RawData", "Press_1027"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Single experiment start time — all CSV timestamps are relative to this
    experiment_start = time.monotonic()

    print_info("Connecting to printer controller...")
    printer_serial = serial.Serial(args.printer_port, 115200)
    print_info("Connecting to force sensor...")
    force_serial = serial.Serial(args.force_port, 115200)
    print_info("Setting up EIT reader thread...")

    q_force = queue.Queue(maxsize=200)
    q_eit = queue.Queue(maxsize=4000)

    force_reader = ForceReader(force_serial, q_force)
    force_reader.start()

    eit_reader = AmodoEITDevice(q_out=q_eit)
    eit_reader.start()
    print_info("EIT and force readers started.")

    measurement_runner = MeasurementRunner(
        printer_serial=printer_serial,
        q_force=q_force,
        q_eit=q_eit,
        experiment_start=experiment_start,
        output_dir=output_dir,
    )
    measurement_runner.start()
    print_info("Measurement runner started.")

    try:
        measurement_runner.join()
    except KeyboardInterrupt:
        print_info("\n\nStopped by user.")
    finally:
        measurement_runner.stop()
        eit_reader.stop()
        force_reader.stop()

        measurement_runner.join(timeout=2.0)
        eit_reader.join(timeout=2.0)
        force_reader.join(timeout=2.0)

        printer_serial.close()
        force_serial.close()
        print_info("All threads stopped and resources cleaned up.")


if __name__ == "__main__":
    main()