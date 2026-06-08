import argparse
import csv
import os
import queue
import threading
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import serial
import pyeit.eit.protocol as protocol
import amodo_eit as eit
from utils import print_info


class AmodoEITDevice(threading.Thread):
    def __init__(self, NUM_ELECTRODES, INJ_STEP, READ_STEP, STIM_FREQ_KHZ, PERIODS_PER_MEASUREMENT, TX_GAIN, RX_GAIN, q_out, group=None):
        super().__init__(group=group, name="AmodoEITDevice", daemon=True)
        self.devices = eit.get_connected_devices()
        self.q = q_out
        self.stop_evt = threading.Event()
        self.stim_freq_khz = STIM_FREQ_KHZ
        self.periods_per_measurement = PERIODS_PER_MEASUREMENT
        self.tx_gain = TX_GAIN
        self.rx_gain = RX_GAIN
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
            parser_meas="rotate_meas"
        )

    def _configure_and_start_streaming(self):
        print_info(f"Using device: {self.device.port}, version {self.device.version}, build {self.device.build_date_time}")
        self.device.set_stimulation_frequency(self.stim_freq_khz)

        print_info("Loading electrode configuration...")
        electrode_configurations = []
        for i_exc, exc_pair in enumerate(self.protocol_obj.ex_mat):
            A, B = exc_pair
            meas_pairs = self.protocol_obj.meas_mat[i_exc]
            for M, N in meas_pairs:
                pin_offset = 1
                configuration = (A + pin_offset, B + pin_offset, M + pin_offset, N + pin_offset, self.tx_gain, self.rx_gain)
                electrode_configurations.append(configuration)

        self.device.set_electrode_configurations(electrode_configurations)
        print_info(f"Electrode configuration loaded ({len(electrode_configurations)} configurations).\n")
        self.device.set_num_periods_to_sample_per_measurement(self.periods_per_measurement)

        print_info("Capturing baseline frame...")
        self.device.start_streaming()

        while self.device.latest_frame is None and not self.stop_evt.is_set():
            time.sleep(0.01)

        if self.device.latest_frame is not None:
            baseline_frame, baseline_clipping = self.device.latest_frame
            baseline_frame = np.array([x if x > 1e-12 else 1e-12 for x in baseline_frame])
            if baseline_clipping:
                print_info("Clipping detected in baseline")
            print_info(f"Baseline captured: {len(baseline_frame)} measurements\n")

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
                        "t_wall": datetime.now().isoformat(),
                        "epoch": time.time(),
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
                    "t_wall": datetime.now().isoformat(),
                    "epoch": time.time(),
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


class LiveEITPlotter(threading.Thread):
    def __init__(self, q_eit, q_force, current_state, refresh_interval=0.1, group=None):
        super().__init__(group=group, name="LiveEITPlotter", daemon=True)
        self.q_eit = q_eit
        self.q_force = q_force
        self.current_state = current_state
        self.refresh_interval = refresh_interval
        self.stop_evt = threading.Event()

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.line, = self.ax.plot([], [], color="tab:blue")
        self.ax.set_xlabel("EIT channel index")
        self.ax.set_ylabel("Value")
        self.ax.grid(False)
        self.ax.set_xticks([])
        self.ax.set_title("Live EIT signal")
        self.fig.canvas.manager.set_window_title("Live EIT signal")
        self.fig.canvas.draw()

    def _peek_latest(self, q):
        with q.mutex:
            if q.queue:
                return q.queue[-1]
        return None

    def _update_title(self, force_value, position):
        if force_value is not None and position is not None:
            return f"Force={force_value:.3f}N | Position={position}"
        if force_value is not None:
            return f"Force={force_value:.3f}N"
        if position is not None:
            return f"Position={position}"
        return "Live EIT signal"

    def run(self):
        while not self.stop_evt.is_set():
            latest_eit = self._peek_latest(self.q_eit)
            latest_force = self._peek_latest(self.q_force)
            if latest_eit is not None and "readings" in latest_eit and latest_eit["readings"]:
                values = np.asarray(latest_eit["readings"], dtype=float)
                x = np.arange(values.size)
                self.line.set_data(x, values)
                self.ax.set_xlim(0, max(1, values.size - 1))
                ymin, ymax = np.min(values), np.max(values)
                margin = max(abs(ymin), abs(ymax), 1e-6) * 0.05
                self.ax.set_ylim(ymin - margin, ymax + margin)

                force_value = latest_force.get("force") if latest_force is not None else self.current_state.get("force")
                position = self.current_state.get("position")
                self.ax.set_title(self._update_title(force_value, position))
                self.fig.canvas.draw_idle()
                plt.pause(0.001)

            time.sleep(self.refresh_interval)

    def stop(self):
        self.stop_evt.set()


class PrinterController:
    def __init__(self, printer_serial, q_out):
        self.printer_serial = printer_serial
        self.q = q_out

    def _enqueue_event(self, event_type, payload, description=None):
        event = {
            "t": time.monotonic(),
            "t_wall": datetime.now().isoformat(),
            "epoch": time.time(),
            "event_type": event_type,
            "payload": payload,
            "description": description,
        }
        try:
            self.q.put_nowait(event)
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            self.q.put_nowait(event)

    def write(self, command, description=None):
        self.printer_serial.write(command)
        command_text = command.decode(errors='ignore') if isinstance(command, (bytes, bytearray)) else str(command)
        self._enqueue_event("printer_command", command_text, description)

    def wait_for_position(self):
        self.printer_serial.flush()
        self.write(str.encode("M114 R\r\n"), description="position_request")
        while True:
            line = self.printer_serial.readline()
            if not line:
                continue
            decoded = line.decode(errors='ignore').strip()
            self._enqueue_event("printer_response", decoded, description="position_response")
            if 'Count' in decoded:
                break

    def close(self):
        try:
            self.printer_serial.close()
        except Exception:
            pass


def sample_eit_now(q_eit):
    last = None
    while True:
        try:
            last = q_eit.get_nowait()
        except queue.Empty:
            break
    return last


def waitforposition(printer):
    printer.wait_for_position()


# Printer-control defaults
Cornerposition = (70, 28, 20)
retractheight = 5
step = 0.05
waittime = 5


def sample_force_now(q_force):
    last = None
    while True:
        try:
            last = q_force.get_nowait()
        except queue.Empty:
            break
    return last


def takereading(targetforce, printer, q_force):
    starttime = datetime.now()
    printer.write(str.encode("G1 Z" + str(Cornerposition[2]) + " F400\r\n"), description="lower_to_start")
    print_info(f"Lowering probe to starting position (Z={Cornerposition[2]}mm) for target force {targetforce}N...")
    printer.wait_for_position()
    print_info("Starting force readings...")
    print_info("Waiting for initial force reading...")

    force_sample = None
    while force_sample is None:
        force_sample = sample_force_now(q_force)
        if force_sample is None:
            time.sleep(0.05)

    force = force_sample["force"]
    n = 1
    print_info(f"Initial force reading: {force:.3f}N. Adjusting position to reach target force of {targetforce}N...")
    while force < targetforce:
        print_info(f"Current force: {force:.3f}N, below target of {targetforce}N. Lowering further...")
        printer.write(str.encode("G1 Z" + str(Cornerposition[2] - n * step) + " F400\r\n"), description="lower_step")
        printer.wait_for_position()

        force_sample = None
        while force_sample is None:
            force_sample = sample_force_now(q_force)
            if force_sample is None:
                time.sleep(0.05)

        force = force_sample["force"]
        print_info(f"Updated force reading: {force:.3f}N")
        n += 1
        if n * step > 2:
            break

    midtime = datetime.now()
    time.sleep(waittime)
    printer.write(str.encode("G1 Z" + str(Cornerposition[2] + retractheight) + " F400\r\n"), description="retract_probe")
    printer.wait_for_position()
    endtime = datetime.now()
    return force, starttime, midtime, endtime


def setup(printer):
    printer.write(str.encode("G1 Z" + str(Cornerposition[2] + retractheight) + " F400\r\n"), description="raise_probe")
    print("Homing")
    printer.write(str.encode("G28\r\n"), description="home")
    print("Homed")
    print("Moving Z")
    printer.write(str.encode("G1 Z" + str(Cornerposition[2] + retractheight) + " F400\r\n"), description="raise_probe_again")
    print("Moving X")
    printer.write(str.encode("G1 X " + str(Cornerposition[0]) + " Y " + str(Cornerposition[1]) + " F1000\r\n"), description="move_xy")
    print("Asking for Position")
    printer.wait_for_position()
    print("Position found")


def removeProbe(printer):
    printer.write(str.encode("G1 X " + str(Cornerposition[0] + 19) + " Y " + str(Cornerposition[1]) + " F800\r\n"), description="move_probe_out")
    printer.wait_for_position()
    printer.write(str.encode("G1 Z" + str(Cornerposition[2] + 1) + " F400\r\n"), description="lift_probe")
    printer.wait_for_position()


def ensure_combined_header(csv_path, channel_count):
    if os.path.exists(csv_path):
        return

    header = [
        'x_mm', 'y_mm', 'target_force_N', 'actual_force_N',
        'printer_start_utc', 'printer_start_epoch',
        'printer_mid_utc', 'printer_mid_epoch',
        'printer_end_utc', 'printer_end_epoch',
        'eit_capture_utc', 'eit_capture_epoch',
        'eit_sample_monotonic', 'eit_clipping',
    ]
    header.extend([f'eit_channel_{i}' for i in range(channel_count)])

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)


def append_combined_row(csv_path, x, y, targetforce, actualforce, starttime, midtime, endtime, eit_sample, channel_count):
    if not os.path.exists(csv_path):
        ensure_combined_header(csv_path, channel_count)

    eit_capture_utc = ''
    eit_capture_epoch = ''
    eit_sample_mono = ''
    eit_clipping = ''
    eit_values = []

    if eit_sample is not None:
        eit_capture_utc = eit_sample.get('t_wall', '')
        eit_capture_epoch = eit_sample.get('epoch', '')
        eit_sample_mono = eit_sample.get('t', '')
        eit_clipping = eit_sample.get('clipping', '')
        eit_values = eit_sample.get('readings', [])

    row = [
        x, y, targetforce, actualforce,
        starttime.isoformat(), starttime.timestamp(),
        midtime.isoformat(), midtime.timestamp(),
        endtime.isoformat(), endtime.timestamp(),
        eit_capture_utc, eit_capture_epoch,
        eit_sample_mono, eit_clipping,
    ]

    for i in range(channel_count):
        if i < len(eit_values):
            row.append(f"{eit_values[i]:.12f}")
        else:
            row.append('')

    with open(csv_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Synchronize EIT samples with printer control timestamps")
    parser.add_argument('--printer-port', default='COM10', help='Serial port for printer controller')
    parser.add_argument('--force-port', default='COM8', help='Serial port for force sensor')
    parser.add_argument('--output-dir', default=None, help='Directory for CSV logs')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_dir is None:
        output_dir = os.path.join(script_dir, 'RawData', 'Press_1027')
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    combined_csv = os.path.join(output_dir, 'repeats_20.csv')
    legacy_csv = os.path.join(output_dir, 'repeats3.txt')

    print('Connecting to printer controller')
    printer_serial = serial.Serial(args.printer_port, 115200)
    print('Connecting to force sensor')
    force_serial = serial.Serial(args.force_port, 115200)
    print('Setting up EIT reader thread')

    q_printer = queue.Queue(maxsize=100)
    q_force = queue.Queue(maxsize=200)
    q_eit = queue.Queue(maxsize=4000)

    current_state = {"position": None, "force": None}
    plotter = LiveEITPlotter(q_eit=q_eit, q_force=q_force, current_state=current_state)

    printer = PrinterController(printer_serial, q_printer)
    force_reader = ForceReader(force_serial, q_force)
    force_reader.start()

    eit_reader = AmodoEITDevice(
        NUM_ELECTRODES=16,
        INJ_STEP=8,
        READ_STEP=1,
        STIM_FREQ_KHZ=50,
        PERIODS_PER_MEASUREMENT=20,
        TX_GAIN=32,
        RX_GAIN=2,
        q_out=q_eit,
    )
    eit_reader.start()
    plotter.start()

    eit_channel_count = 0
    t0 = time.time()
    while time.time() - t0 < 10:
        try:
            samp = q_eit.get(timeout=1.0)
            if samp and 'readings' in samp and len(samp['readings']) > 0:
                eit_channel_count = len(samp['readings'])
                print_info(f"Detected EIT channel count: {eit_channel_count}")
                break
        except queue.Empty:
            continue

    if eit_channel_count == 0:
        print_info("No EIT data detected, exiting")
        eit_reader.stop()
        eit_reader.join(timeout=1.0)
        force_reader.stop()
        force_reader.join(timeout=1.0)
        printer.close()
        force_serial.close()
        return

    print('Connected')
    time.sleep(2)

    try:
        setup(printer)
        print_info("Starting synchronized logging of printer control and EIT data...\n")
        xs = [0,  0, 15, 15, 25, 25]
        ys = [0, 15, 1, 14, 2, 13]
        target = [0.5, 1, 1.5, 2, 3, 4]

        for k in range(len(target)):
            for i in range(3):
                for j in range(len(xs)):
                    x = xs[j]
                    y = ys[j]
                    targetforce = target[k]

                    printer.write(str.encode("G1 X " + str(Cornerposition[0] + x) + " Y " + str(Cornerposition[1] + y) + " F800\r\n"), description="move_xy")
                    printer.wait_for_position()
                    current_state["position"] = f"x={x}mm y={y}mm"
                    print_info(f"Moved to x={x}mm, y={y}mm. Target force: {targetforce}N. Taking reading...")
                    actualforce, starttime, midtime, endtime = takereading(targetforce, printer, q_force)
                    current_force_sample = sample_force_now(q_force)
                    if current_force_sample is not None:
                        current_state["force"] = current_force_sample.get("force")
                    print_info(f"Measurement taken. Start: {starttime.isoformat()}, Mid: {midtime.isoformat()}, End: {endtime.isoformat()}")

                    while not q_eit.empty():
                        try:
                            q_eit.get_nowait()
                            print_info("Cleared old EIT samples from queue")
                        except queue.Empty:
                            print_info("No more old EIT samples to clear")
                            break
                    print_info("Waiting briefly to capture EIT sample corresponding to this measurement...")
                    time.sleep(0.5)
                    eit_sample = sample_eit_now(q_eit)

                    with open(legacy_csv, 'a', newline='') as legacy_file:
                        legacy_file.write(f"{x}, {y}, {targetforce}, {starttime}, {midtime}, {endtime}\n")

                    append_combined_row(
                        combined_csv,
                        x=x,
                        y=y,
                        targetforce=targetforce,
                        actualforce=actualforce,
                        starttime=starttime,
                        midtime=midtime,
                        endtime=endtime,
                        eit_sample=eit_sample,
                        channel_count=eit_channel_count,
                    )

                    print(f"Logged measurement at x={x}, y={y}, force={targetforce} with EIT capture {eit_sample['t_wall'] if eit_sample else 'none'}")
                    time.sleep(waittime)

    finally:
        try:
            removeProbe(printer)
        except Exception:
            pass
        plotter.stop()
        plotter.join(timeout=1.0)
        eit_reader.stop()
        eit_reader.join(timeout=1.0)
        force_reader.stop()
        force_reader.join(timeout=1.0)
        printer.close()
        force_serial.close()

        print(f"Combined synchronized log saved to {combined_csv}")
        print(f"Legacy printer log saved to {legacy_csv}")


if __name__ == '__main__':
    main()
