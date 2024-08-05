# -*- coding: utf-8 -*-
"""
Extracts Trodes camera module timestamps from exported .videoTimeStamps files

Requires Numpy to be installed.
@author: Chris@spikeGadgets.com
Assumes Python 3.9
"""
import re
import numpy as np

def read_timestamps_old(filename):
    """takes filepath for videotimestamps and returns a np array of timestamps"""
    fid = open(filename, "rb") # opens file in binary read mode

    header_text = np.fromfile(fid, dtype=np.int8, count=50)
    header_char = header_text.tobytes().decode("utf-8", errors="ignore")
    try:
        match = re.search(r"Clock rate:\s*(\d+)", header_char)
        if match:
            clock_rate = int(match.group(1))
        else:
            raise ValueError(f"Clock rate not found in the expected format. - {filename}")
    except IndexError:
        print(f"index out of range because header_char - {header_char}")
        print(f"length is {len(header_char)}")
        print(filename)
        print(f"second line {header_char.splitlines()[1]}")

    fid.seek(0)
    _ = np.fromfile(fid, count=50, dtype=np.uint8)
    timestamps = np.fromfile(fid, dtype=np.uint32) / clock_rate
    return timestamps

def read_timestamps(filename):
    """
    Reads the timestamps from a .videoTimeStamps file.

    Parameters:
        filename (str): The name of the .videoTimeStamps file.
        forcetype (str, optional): Forces the datatype. Default is None.

    Returns:
        tuple: (timestamps, clock_rate)
            timestamps (numpy.ndarray): The timestamps of the camera frames.
            clock_rate (int): The clock rate.
    """
    clock_rate = 30000  # default clock rate

    with open(filename, 'rb') as fid:
        header_text = fid.read(49).decode('utf-8')
        end_header_loc = header_text.find('<End settings>')

        if end_header_loc != -1:
            headersize = end_header_loc + 14
            clock_rate_loc = header_text.find('Clock rate:')
            if clock_rate_loc != -1:
                clock_rate_str = header_text[clock_rate_loc + 12:].split()[0]
                clock_rate = int(clock_rate_str)
        else:
            headersize = 0

        fid.seek(headersize)
        _ = fid.read(headersize)

        timestamps = np.fromfile(fid, dtype=np.uint32).astype(np.double) / clock_rate
        print(clock_rate)

    return timestamps

CLOCK_STRING = "Clock rate: "
HEADER_END_STRING = "End settings"


def read_timestamps_new(filename):
    '''
    The below function reads the header in order to get the clock rate, then
    reads the rest of the file as uint32s and divides by the clock rate to get
    the timestamps in seconds.

    The header length switches, so reading lines seems more reliable..
    Encoding appears to be latin-1, not UTF-8.
    '''
    with open(filename, "r", encoding='latin-1') as fid:
        while True:
            header_text = fid.readline()
            # find substring "clock rate: " in header_text
            if header_text.find(CLOCK_STRING) != -1:
                clock_rate = int(re.search(r'\d+', header_text)[0])
            elif header_text.find(HEADER_END_STRING) != -1:
                break
        timestamps = np.fromfile(fid, dtype=np.uint32) / clock_rate
    return timestamps