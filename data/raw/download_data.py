import os
import zipfile
import requests
import pandas as pd

os.makedirs("data/raw", exist_ok=True)

url = "https://www.cdc.gov/brfss/annual_data/2024/files/LLCP2024XPT.zip"
zip_path = "data/raw/LLCP2024XPT.zip"

response = requests.get(url, stream=True)
response.raise_for_status()

with open(zip_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            f.write(chunk)

print("Downloaded")

extract_path = "data/raw/brfss_2024"
os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Extracted files:", os.listdir(extract_path))

xpt_files = [
    file
    for file in os.listdir(extract_path)
    if file.strip().lower().endswith(".xpt")
]

print("XPT files:", xpt_files)

xpt_path = os.path.join(extract_path, xpt_files[0])

df = pd.read_sas(
    xpt_path,
    format="xport",
    encoding="latin1"
)

# Save as Parquet
parquet_path = "data/raw/brfss_2024_raw.parquet"
df.to_parquet(parquet_path, index=False)

print("Saved parquet:", parquet_path)

# Save as CSV
csv_path = "data/raw/brfss_2024_raw.csv"
df.to_csv(csv_path, index=False)

print("Saved csv:", csv_path)