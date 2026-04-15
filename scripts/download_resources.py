"""
Скрипт для скачивания необходимых ресурсов:
- Pre-extracted alignments
- WaveGlow вокодер
"""

import os
import sys
import subprocess


def download_file(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if "drive.google.com" in url:
        try:
            import gdown
            gdown.download(url, output_path, quiet=False)
        except ImportError:
            print("Install gdown: pip install gdown")
            sys.exit(1)
    else:
        subprocess.run(["wget", "-O", output_path, url], check=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--waveglow", action="store_true")
    parser.add_argument("--alignments", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not any([args.all, args.waveglow, args.alignments]):
        args.all = True

    if args.all or args.waveglow:
        print("Downloading WaveGlow...")
        url = "https://api.ngc.nvidia.com/v2/models/nvidia/waveglowpyt_fp32/versions/1/files/nvidia_waveglowpyt_fp32_20190306.pt"
        download_file(url, "vocoder/waveglow.pt")

    if args.all or args.alignments:
        print("Downloading alignments...")
        print("NOTE: Update URL to actual alignments link from course materials")
        # download_file(ALIGNMENTS_URL, "data/alignments.tar.gz")

    print("Done!")
