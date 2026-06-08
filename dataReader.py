import argparse
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a CSV and interactively plot EIT channel values row-by-row."
    )
    parser.add_argument(
        "--csv_file",
        default="RawData\\Press_1027\\repeats_20.csv",
        help="Path to the CSV file to load.",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="Initial row index to display (0-based).",
    )
    return parser.parse_args()


def find_eit_columns(columns):
    pattern = re.compile(r"^eit_channel_(\d+)$")
    matches = []
    for col in columns:
        m = pattern.match(col)
        if m:
            matches.append((int(m.group(1)), col))
    if not matches:
        raise ValueError("No columns matching 'eit_channel_n' found in CSV.")
    return [col for _, col in sorted(matches, key=lambda pair: pair[0])]


def sort_rows_by_xy_then_force(
    df: pd.DataFrame,
    x_col: str = "x_mm",
    y_col: str = "y_mm",
    force_col: str = "target_force_N",
):
    if x_col not in df.columns or y_col not in df.columns or force_col not in df.columns:
        missing = [col for col in (x_col, y_col, force_col) if col not in df.columns]
        raise KeyError(f"Missing required columns for sorting: {missing}")

    return df.sort_values([x_col, y_col, force_col], kind="mergesort").reset_index(drop=True)


def load_data(csv_file: str):
    df = pd.read_csv(csv_file)
    df = sort_rows_by_xy_then_force(df)
    eit_columns = find_eit_columns(df.columns)
    return df, eit_columns



def get_label(row_index: int, row_data: pd.Series) -> str:
    label_parts = [f"row={row_index}"]
    if "target_force_N" in row_data.index:
        label_parts.append(f"force={row_data['target_force_N']}")
    if "x_mm" in row_data.index and "y_mm" in row_data.index:
        label_parts.append(f"xy=({row_data['x_mm']},{row_data['y_mm']})")
    return " | ".join(label_parts)


def plot_row(ax, channel_indices, values, title):
    ax.clear()
    ax.plot(channel_indices, values, color="tab:blue")
    ax.set_xlabel("EIT channel index")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(False)
    ax.set_xticks([])
    ax.figure.tight_layout()


def main() -> int:
    args = parse_args()

    try:
        df, eit_columns = load_data(args.csv_file)
    except Exception as exc:
        print(f"Error loading CSV: {exc}", file=sys.stderr)
        return 1

    if df.shape[0] == 0:
        print("CSV file contains no rows.", file=sys.stderr)
        return 1

    current_row = max(0, min(args.row, len(df) - 1))
    channel_indices = [int(re.search(r"(\d+)$", col).group(1)) for col in eit_columns]

    fig, ax = plt.subplots(figsize=(10, 6))

    def update_plot():
        nonlocal current_row
        row_data = df.iloc[current_row]
        values = row_data[eit_columns].astype(float)
        title = get_label(current_row, row_data)
        plot_row(ax, channel_indices, values.values, title)
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal current_row
        if event.key == "up":
            if current_row < len(df) - 1:
                current_row += 1
                update_plot()
            else:
                print("Already at last row.")
        elif event.key == "down":
            if current_row > 0:
                current_row -= 1
                update_plot()
            else:
                print("Already at first row.")

    update_plot()
    fig.canvas.mpl_connect("key_press_event", on_key)
    print(
        "Use the Up and Down arrow keys in the plot window to navigate rows. "
        "Close the window to exit."
    )
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
