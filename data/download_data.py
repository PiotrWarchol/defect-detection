import urllib.request
import os
import zipfile

print("Setting up manufacturing defect dataset...")

# Use casting defect dataset - freely available, no login required
# Perfect for manufacturing context - detects defects in metal castings
os.makedirs("data", exist_ok=True)

url = "https://github.com/Sachin-PC/Casting-Product-Image-Data-for-Quality-Inspection/archive/refs/heads/master.zip"

print("Downloading casting defect dataset...")
print("This may take a minute...")

urllib.request.urlretrieve(url, "data/casting_data.zip")
print("Download complete!")

print("Extracting files...")
with zipfile.ZipFile("data/casting_data.zip", 'r') as zip_ref:
    zip_ref.extractall("data/")
print("Extraction complete!")

# Check what we got
print("\nDataset structure:")
for root, dirs, files in os.walk("data"):
    level = root.replace("data", '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level < 3:
        subindent = ' ' * 2 * (level + 1)
        print(f'{subindent}{len(files)} files')