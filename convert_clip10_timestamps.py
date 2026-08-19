"""
convert_clip10_timestamps.py — Convert clip-10 AIS CSV timestamps from
datetime string format to epoch milliseconds, matching clip-01 format.

clip-10:  Number,mmsi,lon,lat,...,timestamp
           timestamp = "2022-10-15 08:30:00"

clip-01:  ,mmsi,lon,lat,...,timestamp
           timestamp = 1654315502004
"""
import os
import csv
from datetime import datetime, timezone

AIS_DIR = r"F:\MyWork\Article02\Pfe\projet\data\clips\clip-10\ais"

def convert_timestamp(ts_str):
    """Convert datetime string to epoch milliseconds."""
    ts_str = ts_str.strip()
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp() * 1000)
    except ValueError:
        try:
            return int(ts_str)
        except ValueError:
            return ts_str

def main():
    files = sorted([f for f in os.listdir(AIS_DIR) if f.endswith('.csv')])
    print(f"Found {len(files)} CSV files in {AIS_DIR}")

    converted = 0
    skipped = 0

    for fname in files:
        fpath = os.path.join(AIS_DIR, fname)
        with open(fpath, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) < 2:
            skipped += 1
            continue

        header = rows[0]
        data_rows = rows[1:]

        # Check if already epoch format (first data row timestamp is numeric)
        try:
            int(data_rows[0][-1])
            print(f"  {fname}: already epoch — skipping")
            skipped += 1
            continue
        except (ValueError, IndexError):
            pass

        # Convert timestamps
        new_rows = []
        for row in data_rows:
            if len(row) >= 9:
                row[-1] = str(convert_timestamp(row[-1]))
            new_rows.append(row)

        # Write back with correct header format (empty first column)
        with open(fpath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['', 'mmsi', 'lon', 'lat', 'speed', 'course', 'heading', 'type', 'timestamp'])
            writer.writerows(new_rows)

        converted += 1
        print(f"  {fname}: converted {len(new_rows)} rows")

    print(f"\nDone: {converted} converted, {skipped} skipped")

if __name__ == '__main__':
    main()
