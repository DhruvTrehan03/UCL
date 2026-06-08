import argparse
import csv
import os
import queue
import threading
import time
from datetime import datetime

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


def sample_eit_now(q_eit):
    last = None
    while True:
        try:
            last = q_eit.get_nowait()
        except queue.Empty:
            break
    return last


def waitforposition(Ender):
    Ender.flush()
    Ender.write(str.encode("M114 R\r\n"))
    while True:
        line = Ender.readline()
        if line.find(b'Count') != -1:
            break
    return


# Printer-control defaults
Cornerposition = (70, 28, 20)
retractheight = 5
step = 0.05
waittime = 5


def takereading(targetforce, Ender, Forces):
    starttime = datetime.now()
    Ender.write(str.encode("G1 Z" + str(Cornerposition[2]) + " F400\r\n"))
    print_info(f"Lowering probe to starting position (Z={Cornerposition[2]}mm) for target force {targetforce}N...")
    waitforposition(Ender)
    print_info("Starting force readings...")
    Forces.flushInput()
    print_info("Waiting for initial force reading...")
    while True:
        try:
            force = float(Forces.readline())
            break
        except ValueError:
            print("No force, trying again")

    n = 1
    print_info(f"Initial force reading: {force:.3f}N. Adjusting position to reach target force of {targetforce}N...")
    while force < targetforce:
        print_info(f"Current force: {force:.3f}N, below target of {targetforce}N. Lowering further...")
        Ender.write(str.encode("G1 Z" + str(Cornerposition[2] - n * step) + " F400\r\n"))
        waitforposition(Ender)
        Forces.flushInput()

        while True:
            try:
                force = float(Forces.readline())
                print(force)
                break
            except ValueError:
                print("No force, trying again")

        n += 1
        if n * step > 2:
            break

    midtime = datetime.now()
    time.sleep(waittime)
    Ender.write(str.encode("G1 Z" + str(Cornerposition[2] + retractheight) + " F400\r\n"))
    waitforposition(Ender)
    endtime = datetime.now()
    return force, starttime, midtime, endtime


def setup(Ender):
    Ender.write(str.encode("G1 Z" + str(Cornerposition[2] + retractheight) + " F400\r\n"))
    print("Homing")
    Ender.write(str.encode("G28\r\n"))
    print("Homed")
    print("Moving Z")
    Ender.write(str.encode("G1 Z" + str(Cornerposition[2] + retractheight) + " F400\r\n"))
    print("Moving X")
    Ender.write(str.encode("G1 X " + str(Cornerposition[0]) + " Y " + str(Cornerposition[1]) + " F1000\r\n"))
    print("Asking for Position")
    waitforposition(Ender)
    print("Position found")


def removeProbe(Ender):
    Ender.write(str.encode("G1 X " + str(Cornerposition[0] + 19) + " Y " + str(Cornerposition[1]) + " F800\r\n"))
    waitforposition(Ender)
    Ender.write(str.encode("G1 Z" + str(Cornerposition[2] + 1) + " F400\r\n"))
    waitforposition(Ender)


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
    Ender = serial.Serial(args.printer_port, 115200)
    print('Connecting to force sensor')
    Forces = serial.Serial(args.force_port, 115200)
    print('Setting up EIT reader thread')
    print("Initial Force:", Forces.readline().decode().strip())

    q_eit = queue.Queue(maxsize=4000)
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
        Ender.close()
        Forces.close()
        return

    print('Connected')
    time.sleep(2)

    try:
        setup(Ender)
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

                    Ender.write(str.encode("G1 X " + str(Cornerposition[0] + x) + " Y " + str(Cornerposition[1] + y) + " F800\r\n"))
                    waitforposition(Ender)
                    print_info(f"Moved to x={x}mm, y={y}mm. Target force: {targetforce}N. Taking reading...")
                    actualforce, starttime, midtime, endtime = takereading(targetforce, Ender, Forces)
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
            removeProbe(Ender)
        except Exception:
            pass
        eit_reader.stop()
        eit_reader.join(timeout=1.0)
        Ender.close()
        Forces.close()

        print(f"Combined synchronized log saved to {combined_csv}")
        print(f"Legacy printer log saved to {legacy_csv}")


if __name__ == '__main__':
    main()
