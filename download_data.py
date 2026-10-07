import urllib.request
import re
import os
import zipfile

req = urllib.request.Request('https://movingai.com/benchmarks/mapf/index.html', headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8', 'ignore')

links = re.findall(r'href\s*=\s*["\']([^"\']+)["\']', html)
target_links = [l for l in links if 'random-32-32-20' in l]
print("Links found for random-32-32-20:", target_links)

for link in target_links:
    if link.endswith('.zip') or link.endswith('.scen'):
        file_url = 'https://movingai.com/benchmarks/mapf/' + link if not link.startswith('http') else link
        fname = os.path.join('data', os.path.basename(link))
        print(f"Downloading {file_url} to {fname}...")
        try:
            urllib.request.urlretrieve(file_url, fname)
            print(f"Downloaded {fname}")
            if fname.endswith('.zip'):
                with zipfile.ZipFile(fname, 'r') as z:
                    z.extractall('data')
                print(f"Extracted {fname}")
        except Exception as e:
            print(f"Failed to download {file_url}: {e}")
