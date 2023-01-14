import zipfile
import glob
import os
from pathlib import Path


root = "/home/zeyi/longtail/property_centric"
paths = glob.glob(f"{root}/*/")

for path in paths:
	path = Path(path)
	path_to_zip_file = path / "raw_expanded.zip"
	directory_to_extract_to = f"{path}"
	try:
		with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
			zip_ref.extractall(directory_to_extract_to)
	except:
		continue


for path in paths:
	path = Path(path)
	path_to_zip_file = path / "raw_expanded.zip"

	try:
		os.remove(path_to_zip_file)
	except:
		continue