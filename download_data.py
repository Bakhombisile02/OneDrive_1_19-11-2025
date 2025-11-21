import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("porinitahoque/parkinsons-disease-pd-data-analysis")

print("Path to dataset files:", path)

# Find the csv file in the downloaded path
for file in os.listdir(path):
    if file.endswith(".csv"):
        source = os.path.join(path, file)
        destination = os.path.join(os.getcwd(), "data", "Parkinsons_Speech-Features.csv")
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy(source, destination)
        print(f"Copied {file} to {destination}")
        break
