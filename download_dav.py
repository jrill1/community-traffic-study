import argparse, hashlib, json, os, subprocess, time
from datetime import datetime, date as _date

HOSTS    = {"WF": "http://192.168.50.50", "EF": "http://192.168.50.51"}
USERNAME = "admin"

parser = argparse.ArgumentParser()
parser.add_argument("date", nargs="?", default=None, help="Date in YYYY_MM_DD format (default: today)")
parser.add_argument("cam", choices=["EF", "WF"], help="Camera: EF or WF")
parser.add_argument("--password", help="Camera password (or set CAMERA_PASSWORD env var)")
parser.add_argument("--quiet", action="store_true", help="Print one stats line instead of verbose output")
args = parser.parse_args()

DATE     = (args.date or _date.today().strftime("%Y_%m_%d")).replace("_", "-")
HOST     = HOSTS[args.cam]
DATE_DIR = DATE.replace("-", "_")
DAV_DIR  = f"/Users/jrill/Documents/traffic_project/traffic_data/{DATE_DIR}/{args.cam}/dav"
MP4_DIR  = f"/Users/jrill/Documents/traffic_project/traffic_data/{DATE_DIR}/{args.cam}/mp4"
os.makedirs(DAV_DIR, exist_ok=True)
os.makedirs(MP4_DIR, exist_ok=True)

password = args.password or os.environ.get("CAMERA_PASSWORD") or input("Camera password: ").strip()

def log(msg):
    if not args.quiet:
        print(msg)

def _rpc_raw(endpoint, payload):
    r = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{HOST}/{endpoint}",
         "-H", "Content-Type: application/json",
         "-d", json.dumps(payload)],
        capture_output=True, text=True,
    )
    return json.loads(r.stdout) if r.stdout.strip() else {}


def login():
    r1 = _rpc_raw("RPC2_Login", {
        "method": "global.login",
        "params": {"userName": USERNAME, "password": "", "clientType": "Web5.0"},
        "id": 1,
    })
    p       = r1["params"]
    realm   = p["realm"]
    random_ = p["random"]
    session = r1["session"]

    pwd_md5   = hashlib.md5(f"{USERNAME}:{realm}:{password}".encode()).hexdigest().upper()
    auth_hash = hashlib.md5(f"{USERNAME}:{random_}:{pwd_md5}".encode()).hexdigest().upper()

    r2 = _rpc_raw("RPC2_Login", {
        "method": "global.login",
        "params": {"userName": USERNAME, "password": auth_hash,
                   "clientType": "Web5.0", "authorityType": "Default"},
        "id": 2,
        "session": session,
    })
    if not r2.get("result"):
        print(f"Login failed: {r2}")
        exit(1)
    return r2["session"]


SESSION = login()
COOKIE  = f"DhWebClientSessionID={SESSION}"

def rpc(payload):
    r = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{HOST}/RPC2",
         "-H", "Content-Type: application/json",
         "-H", f"Cookie: {COOKIE}",
         "-d", json.dumps(payload)],
        capture_output=True, text=True,
    )
    return json.loads(r.stdout) if r.stdout.strip() else {}


result = rpc({"method": "mediaFileFind.factory.create", "params": None, "id": 1, "session": SESSION})
if "result" not in result:
    print(f"Error: {result}")
    exit(1)
obj = result["result"]

rpc({"method": "mediaFileFind.findFile", "id": 2, "object": obj, "session": SESSION,
     "params": {"condition": {"Channel": 0, "Dirs": ["/mnt/dvr/mmc2p2_0"],
     "StartTime": f"{DATE} 00:00:00", "EndTime": f"{DATE} 23:59:59",
     "Flags": ["Timing", "Event", "Manual"], "Types": ["dav"],
     "Order": "Ascent", "Redundant": "Exclusion"}}})

all_files = []
while True:
    r = rpc({"method": "mediaFileFind.findNextFile", "id": 3, "object": obj,
             "session": SESSION, "params": {"Count": 100}})
    infos = r.get("params", {}).get("infos")
    if not infos:
        log(f"  No more files. Total: {len(all_files)}")
        break
    all_files.extend(infos)
    log(f"  Fetched {len(all_files)} files so far...")
    if len(infos) < 100:
        break

n = len(all_files)
log(f"\nFound {n} files. Downloading and converting to {MP4_DIR}")

# --- Download + Convert ---
t_start = time.time()
n_dl = 0
n_cv = 0
t_dl = 0.0
t_cv = 0.0
n_skip = 0

for i, f in enumerate(all_files):
    path     = f["FilePath"]
    name     = os.path.basename(path)
    mp4_name = name[:-4] + ".mp4"
    url      = f"{HOST}/RPC_Loadfile{path}"
    dav_dest = os.path.join(DAV_DIR, name)
    mp4_dest = os.path.join(MP4_DIR, mp4_name)

    if os.path.exists(mp4_dest) and os.path.getsize(mp4_dest) > 0:
        n_skip += 1
        continue

    log(f"[{i+1}/{n}] {name}  downloading...")
    t0 = time.time()
    subprocess.run(["curl", "-g", "-s", "-H", f"Cookie: {COOKIE}", url, "-o", dav_dest])
    t_dl += time.time() - t0
    n_dl += 1

    log(f"[{i+1}/{n}] {name}  converting...")
    t0 = time.time()
    subprocess.run(["ffmpeg", "-y", "-i", dav_dest, "-c:v", "libx264", "-c:a", "aac", mp4_dest],
                   capture_output=True)
    t_cv += time.time() - t0
    n_cv += 1

    os.remove(dav_dest)

# --- Output ---
if args.quiet:
    # Single line for combined summary in run_download.sh
    if n_dl:
        print(f"{n_dl} downloaded  ({n_skip} skipped)  dl={t_dl:.0f}s  cv={t_cv:.0f}s")
    else:
        print(f"all {n_skip} skipped")
else:
    n_e2e = n_dl
    t_e2e = t_dl + t_cv
    print(f"\n{'─'*46}")
    if n_skip:
        print(f"Skipped  :  {n_skip:4} files  (already downloaded)")
    print(f"Download :  {n_dl:4} files  {t_dl:7.1f}s total  {t_dl/n_dl:6.1f}s avg" if n_dl else "Download :     0 files")
    print(f"Convert  :  {n_cv:4} files  {t_cv:7.1f}s total  {t_cv/n_cv:6.1f}s avg" if n_cv else "Convert  :     0 files")
    print(f"E2E      :  {n_e2e:4} files  {t_e2e:7.1f}s total  {t_e2e/n_e2e:6.1f}s avg" if n_e2e else "E2E      :     0 files")
    print(f"Last ran at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*46}")
