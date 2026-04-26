import os

import pynmrstar


def download_bmrb_rdcs(bmrb_id: str, output_path: str) -> None:
    """Download BMRB data and save to a file."""
    if os.path.exists(output_path):
        print(f"Skipping download, {output_path} already exists.")
        return

    print(f"Downloading BMRB entry {bmrb_id}...")
    entry = pynmrstar.Entry.from_database(bmrb_id)
    entry.write_to_file(output_path)
    print(f"Successfully downloaded to {output_path}!")


if __name__ == "__main__":
    download_bmrb_rdcs("6457", "bmrb_6457.str")
