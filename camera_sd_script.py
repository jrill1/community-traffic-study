export CAMERA_PASSWORD=yourpassword

venv311/bin/python3 - WF <<'EOF'
import argparse, hashlib, json, os, subprocess
from collections import Counter

HOSTS = {"WF": "http://192.168.50.50", "EF": "http://192.168.50.51"}
cam = "WF"  # change to EF for the other cam
HOST = HOSTS[cam]
USERNAME = "admin"
password = os.environ.get("CAMERA_PASSWORD") or input("Password: ").strip()

def _rpc_raw(ep, payload):
    r = subprocess.run(["curl","-s","-X","POST",f"{HOST}/{ep}",
        "-H","Content-Type: application/json","-d",json.dumps(payload)],
        capture_output=True, text=True)
    return json.loads(r.stdout) if r.stdout.strip() else {}

r1 = _rpc_raw("RPC2_Login", {"method":"global.login","params":{"userName":USERNAME,"password":"","clientType":"Web5.0"},"id":1})
p = r1["params"]; realm = p["realm"]; rnd = p["random"]; sess = r1["session"]
pwd_md5   = hashlib.md5(f"{USERNAME}:{realm}:{password}".encode()).hexdigest().upper()
auth_hash = hashlib.md5(f"{USERNAME}:{rnd}:{pwd_md5}".encode()).hexdigest().upper()
r2 = _rpc_raw("RPC2_Login", {"method":"global.login","params":{"userName":USERNAME,"password":auth_hash,"clientType":"Web5.0","authorityType":"Default"},"id":2,"session":sess})
sess = r2["session"]
cookie = f"DhWebClientSessionID={sess}"

def rpc(payload):
    r = subprocess.run(["curl","-s","-X","POST",f"{HOST}/RPC2",
        "-H","Content-Type: application/json","-H",f"Cookie: {cookie}",
        "-d",json.dumps(payload)], capture_output=True, text=True)
    return json.loads(r.stdout) if r.stdout.strip() else {}

obj = rpc({"method":"mediaFileFind.factory.create","params":None,"id":1,"session":sess})["result"]
rpc({"method":"mediaFileFind.findFile","id":2,"object":obj,"session":sess,
     "params":{"condition":{"Channel":0,"Dirs":["/mnt/dvr/mmc2p2_0"],
     "StartTime":"2026-01-01 00:00:00","EndTime":"2026-12-31 23:59:59",
     "Flags":["Timing","Event","Manual"],"Types":["dav"],"Order":"Ascent","Redundant":"Exclusion"}}})

dates = Counter()
while True:
    r = rpc({"method":"mediaFileFind.findNextFile","id":3,"object":obj,"session":sess,"params":{"Count":100}})
    infos = r.get("params",{}).get("infos")
    if not infos: break
    for f in infos:
        dates[f["StartTime"][:10]] += 1
    if len(infos) < 100: break

for d, n in sorted(dates.items()):
    print(f"  {d}  {n} clips")
EOF
