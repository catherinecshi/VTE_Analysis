# -*- coding: utf-8 -*-
"""
Extracts Trodes camera module timestamps from exported .videoTimeStamps files

Requires Numpy to be installed.
@author: Chris@spikeGadgets.com
Assumes Python 3.9
"""
import numpy as np


def read_timestamps(filename):
    """takes filepath for videotimestamps and returns a np array of timestamps"""
    fid = open(filename, "rb") # opens file in binary read mode

    header_text = np.fromfile(fid, dtype=np.int8, count=50)
    header_char = header_text.tobytes().decode("utf-8", errors="ignore")
    clock_rate = int(header_char.splitlines()[1][12:])

    fid.seek(0)
    junk = np.fromfile(fid, count=50, dtype=np.uint8)
    timestamps = np.fromfile(fid, dtype=np.uint32) / clock_rate
    return timestamps