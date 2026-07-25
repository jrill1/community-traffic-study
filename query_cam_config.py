import hashlib, json, os, subprocess, sys

HOSTS    = {"WF": "http://192.168.50.50", "EF": "http://192.168.50.51"}
USERNAME = "admin"
password = os.environ.get("CAMERA_PASSWORD") or input("Camera password: ").strip()

CONFIGS = [
    "VideoColor",
    "Exposure",
    "BackLight",
    "Backlight",
    "WhiteBalance",
    "DayNight",
    "Dehaze",
    "NightOptions",
    "CameraParam",
    "VideoInOptions",
    "AutoExposure",
    "ImageCustom",
    "VideoStandard",
    "VideoInExposure",
    "DNNight",
    "NrParam",
    "VideoInDenoise",
]

def _post(host, endpoint, payload):
    r = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{host}/{endpoint}",
         "-H", "Content-Type: application/json",
         "-d", json.dumps(payload)],
        capture_output=True, text=True,
    )
    return json.loads(r.stdout) if r.stdout.strip() else {}

def login(host):
    r1 = _post(host, "RPC2_Login", {
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
    r2 = _post(host, "RPC2_Login", {
        "method": "global.login",
        "params": {"userName": USERNAME, "password": auth_hash,
                   "clientType": "Web5.0", "authorityType": "Default"},
        "id": 2, "session": session,
    })
    if not r2.get("result"):
        print(f"Login failed for {host}: {r2}")
        sys.exit(1)
    return r2["session"]

def rpc(host, session, payload):
    cookie = f"DhWebClientSessionID={session}"
    r = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{host}/RPC2",
         "-H", "Content-Type: application/json",
         "-H", f"Cookie: {cookie}",
         "-d", json.dumps(payload)],
        capture_output=True, text=True,
    )
    return json.loads(r.stdout) if r.stdout.strip() else {}

def get_config(host, session, name):
    return rpc(host, session, {
        "method": "configManager.getConfig",
        "params": {"name": name},
        "id": 1, "session": session,
    })

def set_config(host, session, name, table):
    return rpc(host, session, {
        "method": "configManager.setConfig",
        "params": {"name": name, "table": table},
        "id": 1, "session": session,
    })

TEST_MODE = "--test" in sys.argv

if TEST_MODE:
    print("\n=== TEST: Set WF VideoColor Day Brightness 50 -> 60 ===")
    wf_host = HOSTS["WF"]
    wf_session = login(wf_host)

    r = get_config(wf_host, wf_session, "VideoColor")
    table = r["params"]["table"]
    day = table[0][0]
    print(f"Before: {day['Name']} Brightness = {day['Brightness']}")

    day["Brightness"] = 100
    result = set_config(wf_host, wf_session, "VideoColor", table)
    print(f"setConfig result: {result.get('result')}  error: {result.get('error')}")

    r2 = get_config(wf_host, wf_session, "VideoColor")
    print(f"After:  {r2['params']['table'][0][0]['Name']} Brightness = {r2['params']['table'][0][0]['Brightness']}")
else:
    for cam, host in HOSTS.items():
        print(f"\n{'='*60}")
        print(f"  {cam}  ({host})")
        print(f"{'='*60}")
        session = login(host)
        for config in CONFIGS:
            result = get_config(host, session, config)
            if result.get("result"):
                print(f"\n-- {config} --")
                print(json.dumps(result["params"]["table"], indent=2))
            else:
                print(f"\n-- {config} -- (not supported or empty)")
