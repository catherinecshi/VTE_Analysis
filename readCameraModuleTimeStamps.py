# -*- coding: utf-8 -*-
"""
Extracts Trodes camera module timestamps from exported .videoTimeStamps files

Requires Numpy to be installed.
@author: Chris@spikeGadgets.com
Assumes Python 3.9
"""
import numpy as np


def read_timestamps(filename):
    fid = open(filename, "rb")

    headerText = np.fromfile(fid, dtype=np.int8, count=50)
    headerChar = headerText.tobytes().decode("utf-8")
    clockRate = int(headerChar.splitlines()[1][12:])

    fid.seek(0)
    junk = np.fromfile(fid, count=50, dtype=np.uint8)
    timestamps = np.fromfile(fid, dtype=np.uint32) / clockRate
    return timestamps