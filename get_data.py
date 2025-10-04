# get_data.py
# Author: Gabriel Caballero
import os, requests
URL="https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
os.makedirs("data", exist_ok=True)
r=requests.get(URL,timeout=30); r.raise_for_status()
with open("data/names.txt","wb") as f: f.write(r.content)
print("Downloaded names.txt")
