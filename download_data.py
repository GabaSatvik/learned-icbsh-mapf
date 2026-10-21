import urllib.request
import re
import os
import zipfile
import shutil
import glob

os.makedirs('data', exist_ok=True)
maps_to_get = ['random-32-32-20', 'room-32-32-4']

req = urllib.request.Request('https://movingai.com/benchmarks/mapf/index.html', headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req).read().decode('utf-8', 'ignore')
    links = re.findall(r'href\s*=\s*["\']([^"\']+)["\']', html)
except Exception as e:
    print(f"Warning fetching index: {e}")
    links = []

for m in maps_to_get:
    target_links = [l for l in links if m in l]
    if not target_links:
        # Fallback to standard URLs
        target_links = [
            f"{m}.map.zip",
            f"{m}.map-scen-random.zip",
            f"{m}.map-scen-even.zip"
        ]
    print(f"Links found for {m}:", target_links)
    for link in target_links:
        if link.endswith('.zip') or link.endswith('.scen'):
            file_url = 'https://movingai.com/benchmarks/mapf/' + link if not link.startswith('http') else link
            fname = os.path.join('data', os.path.basename(link))
            if not os.path.exists(fname):
                print(f"Downloading {file_url} to {fname}...")
                try:
                    urllib.request.urlretrieve(file_url, fname)
                    print(f"Downloaded {fname}")
                except Exception as e:
                    print(f"Failed to download {file_url}: {e}")
            if os.path.exists(fname) and fname.endswith('.zip'):
                try:
                    with zipfile.ZipFile(fname, 'r') as z:
                        z.extractall('data')
                except Exception as e:
                    print(f"Extract error {fname}: {e}")

# Ensure scenarios are in data/ as well as data/scen-random/
for f in glob.glob('data/scen-random/*.scen'):
    dest = os.path.join('data', os.path.basename(f))
    if not os.path.exists(dest):
        shutil.copy(f, dest)

print("Data download and extraction complete.")
