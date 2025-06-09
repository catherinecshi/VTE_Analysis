"""utilities for simple manipulations with files"""

import csv
from typing import Dict, Optional

def add_row_to_csv(file_path: str, row_data: Dict, headers: Optional[list[str]] = None):
    """
    appends a single row to a csv file
    creates the files with the headers provided if it doesn't exist already

    Parameters:
    - file_path: file path of the csv file
    - row_data: dictionary with {column header: value}
    - headers: list of headers - shoudl correspond to dict. Defaults to None.
    """
    # infer headers if not provided
    if headers is None:
        headers = list(row_data.keys())
    
    # check if the file exists
    file_exists = False
    try:
        with open(file_path, "r", encoding="utf-8"):
            file_exists = True
    except FileNotFoundError:
        pass
    
    with open(file_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        
        # write headers if file is being crated
        if not file_exists and headers:
            writer.writeheader()
        
        writer.writerow(row_data)