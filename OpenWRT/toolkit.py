import modal
import os
import sys
import urllib.request
import re
import glob
import subprocess
import shutil

# Define the Modal App
app = modal.App("openwrt-toolkit")

# Ensure the custom packages directory exists at load time
local_pkgs_dir = os.path.join(os.path.dirname(__file__), "Packages")
os.makedirs(local_pkgs_dir, exist_ok=True)

# Define the container environment and install build dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "build-essential",
        "libncurses-dev",
        "zlib1g-dev",
        "gawk",
        "git",
        "gettext",
        "libssl-dev",
        "xsltproc",
        "wget",
        "unzip",
        "python3",
        "python3-distutils",
        "rsync",
        "file",
        "curl",
        "time",
        "zstd"
    )
)

# Define an Alpine Linux image for ultra-fast IPK to APK conversion
alpine_image = (
    modal.Image.from_registry("alpine:edge")
    .run_commands("apk update && apk add python3 py3-pip apk-tools tar gzip")
)

# The gigantic list of 339 packages extracted from the router
# We append 'luci-theme-argon' and 'luci-app-argon-config' at the end to include the custom ones
PACKAGES = (
    "adb adblock apk-mbedtls attendedsysupgrade-common attr avahi-dbus-daemon base-files block-mount busybox ca-bundle "
    "cgi-io cloudflared collectd collectd-mod-cpu collectd-mod-interface collectd-mod-iwinfo collectd-mod-load "
    "collectd-mod-memory collectd-mod-network collectd-mod-rrdtool coreutils coreutils-sort dbus ddns-scripts "
    "ddns-scripts-services dnsmasq dosfstools dropbear e2fsprogs etherwake fdisk firewall4 fstools fwtool gawk getrandom "
    "hostapd-common icu78 ip-tiny iptables-mod-conntrack-extra iptables-mod-ipopt iptables-nft iw jansson4 jshn jsonfilter "
    "kernel kmod-cfg80211 kmod-crypto-aead kmod-crypto-authenc kmod-crypto-ccm kmod-crypto-cmac kmod-crypto-crc32c "
    "kmod-crypto-ctr kmod-crypto-des kmod-crypto-gcm kmod-crypto-geniv kmod-crypto-gf128 kmod-crypto-ghash kmod-crypto-hash "
    "kmod-crypto-hmac kmod-crypto-hw-eip93 kmod-crypto-manager kmod-crypto-md5 kmod-crypto-null kmod-crypto-rng "
    "kmod-crypto-seqiv kmod-crypto-sha1 kmod-crypto-sha256 kmod-crypto-sha3 kmod-crypto-sha512 kmod-fs-ext4 "
    "kmod-gpio-button-hotplug kmod-hwmon-core kmod-i2c-core kmod-ifb kmod-ipt-conntrack kmod-ipt-conntrack-extra kmod-ipt-core "
    "kmod-ipt-ipopt kmod-leds-gpio kmod-lib-crc-ccitt kmod-lib-crc16 kmod-lib-crc32c kmod-mac80211 kmod-mt76-connac "
    "kmod-mt76-core kmod-mt7603 kmod-mt7615-common kmod-mt7615-firmware kmod-mt7615e kmod-nf-conncount kmod-nf-conntrack "
    "kmod-nf-conntrack6 kmod-nf-flow kmod-nf-ipt kmod-nf-log kmod-nf-log6 kmod-nf-nat kmod-nf-reject kmod-nf-reject6 "
    "kmod-nfnetlink kmod-nft-compat kmod-nft-core kmod-nft-fib kmod-nft-nat kmod-nft-offload kmod-nls-base kmod-ppp "
    "kmod-pppoe kmod-pppox kmod-sched-connmark kmod-sched-core kmod-scsi-core kmod-slhc kmod-usb-common kmod-usb-core "
    "kmod-usb-ledtrig-usbport kmod-usb-storage kmod-usb-storage-uas kmod-usb-xhci-hcd kmod-usb-xhci-mtk kmod-usb3 "
    "libatomic1 libattr libavahi-client libavahi-dbus-support libblkid1 libblobmsg-json20260213 libbz2-1.0 libc libcap "
    "libcap-ng libcomerr0 libcurl4 libdaemon libdbus libdeflate libdht libe2p2 libevent2-7 libevent2-core7 "
    "libevent2-pthreads7 libexpat libext2fs2 libfdisk1 libffi libgcc1 libgdbm libgmp10 libgnutls libidn2 libiptext-nft0 "
    "libiptext0 libiptext6-0 libiwinfo-data libiwinfo20230701 libjson-c5 libjson-script20260213 libltdl7 liblucihttp-ucode "
    "liblucihttp0 liblzma libmbedtls21 libminiupnpc libmnl0 libnatpmp1 libncurses6 libnettle8 libnftnl11 libnghttp2-14 "
    "libnl-tiny1 libopenssl3 libpam libpopt0 libpsl5 libpthread libpython3-3.13 libreadline8 librrd1 librt libsmartcols1 "
    "libsqlite3-0 libss2 libstdcpp6 libtasn1 libtirpc libubox20260213 libubus20251202 libuci20250120 libuclient20201210 "
    "libucode20230711 libudebug libunistring libustream-mbedtls20201210 libutp libuuid1 libuv1 libwebsockets-full libxtables12 "
    "logd luci luci-app-acl luci-app-adblock luci-app-attendedsysupgrade luci-app-cloudflared luci-app-commands luci-app-ddns "
    "luci-app-filemanager luci-app-firewall luci-app-package-manager luci-app-qos luci-app-samba4 luci-app-sshtunnel "
    "luci-app-statistics luci-app-tor luci-app-transmission luci-app-ttyd luci-app-upnp luci-app-watchcat luci-app-wifihistory "
    "luci-app-wol luci-base luci-lib-uqr luci-light luci-mod-admin-full luci-mod-network luci-mod-status luci-mod-system "
    "luci-proto-ipv6 luci-proto-ppp luci-proto-relay luci-theme-bootstrap luci-theme-material miniupnpd-nftables mtd netifd "
    "nftables-json odhcp6c odhcpd-ipv6only openwrt-keyring perl perlbase-base perlbase-bytes perlbase-class perlbase-config "
    "perlbase-dynaloader perlbase-errno perlbase-essential perlbase-fcntl perlbase-filehandle perlbase-getopt perlbase-io "
    "perlbase-list perlbase-net perlbase-posix perlbase-scalar perlbase-selectsaver perlbase-socket perlbase-symbol perlbase-tie "
    "perlbase-time perlbase-xsloader ppp ppp-mod-pppoe procd procd-seccomp procd-ujail python3 python3-asyncio python3-base "
    "python3-codecs python3-ctypes python3-dbm python3-decimal python3-email python3-light python3-logging python3-lzma "
    "python3-multiprocessing python3-ncurses python3-openssl python3-pydoc python3-readline python3-sqlite3 python3-unittest "
    "python3-urllib python3-uuid python3-xml qos-scripts relayd resolveip rpcd rpcd-mod-file rpcd-mod-iwinfo rpcd-mod-luci "
    "rpcd-mod-rpcsys rpcd-mod-rrdns rpcd-mod-ucode rrdtool1 samba4-libs samba4-server sshtunnel tc-tiny terminfo tor tor-hs "
    "transmission-daemon ttyd ubi-utils ubox ubus ubusd uci uclient-fetch ucode ucode-mod-digest ucode-mod-fs ucode-mod-html "
    "ucode-mod-log ucode-mod-math ucode-mod-nl80211 ucode-mod-rtnl ucode-mod-ubus ucode-mod-uci ucode-mod-uloop UDPspeeder "
    "uhttpd uhttpd-mod-ubus urandom-seed urngd usign wakeonlan watchcat wifi-scripts wireless-regdb wpad-basic-mbedtls "
    "xtables-nft zlib luci-theme-argon luci-app-argon-config"
)

@app.function(image=image, cpu=16, memory=16384, timeout=3600)
def compile_package(git_url: str):
    # 1. Dynamically scrape the latest Snapshot SDK URL for ramips/mt7621
    base_url = "https://downloads.openwrt.org/snapshots/targets/ramips/mt7621/"
    print("Fetching latest SDK URL...")
    req = urllib.request.Request(base_url)
    with urllib.request.urlopen(req) as response:
        html = response.read().decode('utf-8')
    
    # Regex to find the SDK tarball
    match = re.search(r'href="(openwrt-sdk-ramips-mt7621_[^"]+\.tar\.zst)"', html)
    if not match:
        raise Exception("Could not find the SDK tarball on the OpenWRT snapshot page.")
    
    sdk_filename = match.group(1)
    sdk_url = base_url + sdk_filename
    
    workspace = "/tmp/workspace"
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    os.makedirs(workspace, exist_ok=True)
    
    # 2. Download and extract SDK (We use /tmp because Modal Volumes don't support hardlinks)
    print(f"Downloading SDK from {sdk_url} ...")
    subprocess.run(["wget", "-q", sdk_url, "-O", f"{workspace}/{sdk_filename}"], check=True)
    
    print("Extracting SDK (this supports hardlinks natively)...")
    subprocess.run(["tar", "-xf", f"{workspace}/{sdk_filename}", "-C", workspace], check=True)
    
    sdk_dir_name = sdk_filename.replace(".tar.zst", "")
    sdk_dir = os.path.join(workspace, sdk_dir_name)
    
    # 3. Clone the custom package git repo into the SDK package/ folder
    print(f"Cloning source code from {git_url} ...")
    package_dir = os.path.join(sdk_dir, "package", "custom_package")
    subprocess.run(["git", "clone", git_url, package_dir], check=True)
    
    # 4. Update and install feeds (required for dependencies like luci.mk)
    print("Updating OpenWRT feeds...")
    subprocess.run(["./scripts/feeds", "update", "-a"], cwd=sdk_dir, check=True)
    subprocess.run(["./scripts/feeds", "install", "-a"], cwd=sdk_dir, check=True)
    
    # 5. Configure and Build the package
    print("Configuring SDK...")
    subprocess.run(["make", "defconfig"], cwd=sdk_dir, check=True)
    
    print("Scanning for packages to compile...")
    make_targets = []
    for root, dirs, files in os.walk(package_dir):
        if 'Makefile' in files:
            # We only care about OpenWRT Makefiles, which define a package.
            # Get the relative path from the SDK root, e.g., 'package/custom_package' or 'package/custom_package/oasis-tool-wireguard'
            rel_path = os.path.relpath(root, sdk_dir)
            make_targets.append(f"{rel_path}/compile")
            
    if not make_targets:
        raise Exception("Could not find any OpenWRT Makefiles in the cloned repository!")
        
    print(f"Found {len(make_targets)} package(s) to compile.")
    print("Compiling across 16 CPU cores (this may take a few minutes)...")
    
    # First attempt: Multi-threaded for speed
    cmd = ["make"] + make_targets + ["-j16"]
    result = subprocess.run(cmd, cwd=sdk_dir)
    
    # If the parallel build fails (common with some OpenWRT packages), fallback to single-threaded verbose
    if result.returncode != 0:
        print("\nParallel build encountered an error, falling back to single-threaded verbose build for debugging...")
        cmd_fallback = ["make"] + make_targets + ["V=s", "-j1"]
        subprocess.run(cmd_fallback, cwd=sdk_dir, check=True)
    
    # 6. Locate the compiled .apk files
    print("Searching for compiled .apk files...")
    # The new apk packages are typically stored in bin/packages/
    apk_files = glob.glob(f"{sdk_dir}/bin/packages/**/*.apk", recursive=True)
    
    if not apk_files:
        raise Exception("Compilation finished, but no .apk files were found in the output directory. Did it compile successfully?")
    
    # Read the files into memory to return them to the local machine
    output_files = {}
    for apk_path in apk_files:
        filename = os.path.basename(apk_path)
        with open(apk_path, "rb") as f:
            output_files[filename] = f.read()
            
    # Cleanup temporary workspace
    shutil.rmtree(workspace)
            
    return output_files


builder_image = image.add_local_dir(
    local_path=local_pkgs_dir,
    remote_path="/Packages"
)

@app.function(image=builder_image, cpu=16, memory=16384, timeout=3600)
def build_firmware():
    # 1. Scrape the latest ImageBuilder URL for ramips/mt7621
    base_url = "https://downloads.openwrt.org/snapshots/targets/ramips/mt7621/"
    print("Fetching latest ImageBuilder URL...")
    req = urllib.request.Request(base_url)
    with urllib.request.urlopen(req) as response:
        html = response.read().decode('utf-8')
    
    # Regex to find the ImageBuilder tarball (using .tar.zst)
    match = re.search(r'href="(openwrt-imagebuilder-ramips-mt7621[^"]+\.tar\.zst)"', html)
    if not match:
        raise Exception("Could not find the ImageBuilder tarball on the OpenWRT snapshot page.")
    
    ib_filename = match.group(1)
    ib_url = base_url + ib_filename
    
    workspace = "/tmp/workspace"
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    os.makedirs(workspace, exist_ok=True)
    
    # 2. Download and extract ImageBuilder
    print(f"Downloading ImageBuilder from {ib_url} ...")
    subprocess.run(["wget", "-q", ib_url, "-O", f"{workspace}/{ib_filename}"], check=True)
    
    print("Extracting ImageBuilder...")
    subprocess.run(["tar", "-xf", f"{workspace}/{ib_filename}", "-C", workspace], check=True)
    
    ib_dir_name = ib_filename.replace(".tar.zst", "")
    ib_dir = os.path.join(workspace, ib_dir_name)
    
    # 3. Inject the custom APK files into the ImageBuilder's packages directory
    print("Injecting custom mounted .apk packages...")
    pkgs_dir = os.path.join(ib_dir, "packages")
    os.makedirs(pkgs_dir, exist_ok=True)
    
    custom_apks = glob.glob("/Packages/*.apk")
    if custom_apks:
        for apk in custom_apks:
            shutil.copy(apk, pkgs_dir)
            print(f"  -> Added {os.path.basename(apk)} to build pool")
    else:
        print("  -> No custom .apk files found in /Packages. Proceeding with standard repo only.")
    
    # 4. Generate custom files for ImageBuilder
    print("Generating automated first-boot configuration scripts...")
    files_dir = os.path.join(ib_dir, "files")
    uci_dir = os.path.join(files_dir, "etc", "uci-defaults")
    os.makedirs(uci_dir, exist_ok=True)

    # 4a. Network, WiFi, and root password setup (runs first)
    net_script = os.path.join(uci_dir, "02-network-setup")
    with open(net_script, "w") as f:
        f.write('''#!/bin/sh

# log potential errors
exec >>/tmp/setup.log 2>&1

wlan_name="Imagination Station"
wlan_name_5g="Imagination Station 5Ghz"
wlan_password="masturbation"
root_password="14235"
lan_ip_address="192.168.2.1"

# Set root password
if [ -n "$root_password" ]; then
  (echo "$root_password"; sleep 1; echo "$root_password") | passwd > /dev/null
fi

# Configure LAN IP
if [ -n "$lan_ip_address" ]; then
  uci set network.lan.ipaddr="$lan_ip_address"
  uci commit network
fi

# Configure WLAN (2.4GHz and 5GHz)
if [ -n "$wlan_name" -a -n "$wlan_password" -a ${#wlan_password} -ge 8 ]; then

  # Enable and configure 2.4GHz (radio0)
  uci set wireless.@wifi-device[0].disabled='0'
  uci set wireless.@wifi-iface[0].disabled='0'
  uci set wireless.@wifi-iface[0].encryption='psk2'
  uci set wireless.@wifi-iface[0].ssid="$wlan_name"
  uci set wireless.@wifi-iface[0].key="$wlan_password"

  # Enable and configure 5GHz (radio1)
  if [ -n "$wlan_name_5g" ]; then
    uci set wireless.@wifi-device[1].disabled='0'
    uci set wireless.@wifi-iface[1].disabled='0'
    uci set wireless.@wifi-iface[1].encryption='psk2'
    uci set wireless.@wifi-iface[1].ssid="$wlan_name_5g"
    uci set wireless.@wifi-iface[1].key="$wlan_password"
  fi

  uci commit wireless
fi

exit 0
''')
    os.chmod(net_script, 0o755)
    print("  -> 02-network-setup: WiFi SSIDs, passwords, LAN IP configured.")

    # 5. Generate the firmware image
    print(f"Building firmware image for Netgear R6900v2 with {len(PACKAGES.split())} packages...")
    
    # Run the make image command
    cmd = [
        "make",
        "image",
        "PROFILE=netgear_r6900-v2",
        f"PACKAGES={PACKAGES}",
        "FILES=files"
    ]
    
    result = subprocess.run(cmd, cwd=ib_dir)
    if result.returncode != 0:
        raise Exception("Firmware compilation failed. Check the logs above.")
        
    # 5. Retrieve the compiled firmware binaries
    print("Searching for generated firmware images...")
    firmware_files = glob.glob(f"{ib_dir}/bin/targets/ramips/mt7621/*sysupgrade.bin")
    firmware_files.extend(glob.glob(f"{ib_dir}/bin/targets/ramips/mt7621/*factory.bin"))
    
    if not firmware_files:
        raise Exception("Firmware built successfully, but no .bin files were found in the output directory.")
    
    # Read files to memory to return to host
    output_files = {}
    for fw_path in firmware_files:
        filename = os.path.basename(fw_path)
        with open(fw_path, "rb") as f:
            output_files[filename] = f.read()
            
    # Cleanup
    shutil.rmtree(workspace)
            
    return output_files


@app.function(image=alpine_image, cpu=1, memory=1024, timeout=300)
def convert_ipk_to_apk(ipk_data: bytes, filename: str):
    import os
    import tarfile
    import subprocess
    import shutil
    import time
    import re
    
    workspace = "/tmp/workspace_apk"
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    os.makedirs(workspace, exist_ok=True)
    
    ipk_path = os.path.join(workspace, filename)
    with open(ipk_path, "wb") as f:
        f.write(ipk_data)
        
    # Extract IPK
    subprocess.run(["tar", "-xzf", ipk_path, "-C", workspace], check=True)
    
    # Extract data.tar.gz and control.tar.gz
    data_dir = os.path.join(workspace, "ipk-extracted")
    control_dir = os.path.join(workspace, "ipk-control")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(control_dir, exist_ok=True)
    
    subprocess.run(["tar", "-xzf", os.path.join(workspace, "data.tar.gz"), "-C", data_dir], check=True)
    subprocess.run(["tar", "-xzf", os.path.join(workspace, "control.tar.gz"), "-C", control_dir], check=True)
    
    # Parse control
    control_path = os.path.join(control_dir, "control")
    metadata = {}
    with open(control_path, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip().lower()] = val.strip()
                
    # --- NEW VERSION SANITIZATION CODE ---
    raw_version = metadata.get('version', '1.0.0')
    
    # 1. Extract the main numeric part (e.g., 2.19.5, or 25.051)
    num_match = re.search(r'([0-9]+(?:\.[0-9]+)*)', raw_version)
    base_version = num_match.group(1) if num_match else "1.0.0"
    
    # 2. Extract any revision number (e.g., from -1 or -r1)
    rev_match = re.search(r'-(?:r)?([0-9]+)', raw_version)
    revision = f"-r{rev_match.group(1)}" if rev_match else "-r0"
    
    # 3. Combine into a guaranteed Alpine-compliant string
    clean_version = f"{base_version}{revision}"
    # -------------------------------------

    # Build apk mkpkg command using the clean_version
    output_apk_name = f"{metadata.get('package', 'unknown')}_{clean_version}_{metadata.get('architecture', 'all')}.apk"
    output_apk_path = os.path.join(workspace, output_apk_name)
    
    cmd = [
        "apk", "mkpkg", 
        "--output", output_apk_path,
        "--info", f"name:{metadata.get('package', 'unknown')}",
        "--info", f"version:{clean_version}",
        "--info", f"description:{metadata.get('description', 'Converted from IPK')}",
        "--info", f"arch:{metadata.get('architecture', 'all')}"
    ]
    
    if 'maintainer' in metadata:
        cmd.extend(["--info", f"maintainer:{metadata['maintainer']}"])
        
    if 'depends' in metadata:
        deps = [d.strip() for d in metadata['depends'].split(',')]
        for dep in deps:
            dep_name = dep.split(' ')[0]
            cmd.extend(["--info", f"depends:{dep_name}"])
            
    cmd.extend(["--files", data_dir])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"apk mkpkg failed:\\nSTDOUT: {result.stdout}\\nSTDERR: {result.stderr}")
        
    with open(output_apk_path, "rb") as f:
        apk_bytes = f.read()
        
    shutil.rmtree(workspace)
    return output_apk_name, apk_bytes


def run_ipk_converter():
    ipk_path = input("\\nEnter the full path to the .ipk file:\\n> ").strip()
    
    if ipk_path.startswith('"') and ipk_path.endswith('"'):
        ipk_path = ipk_path[1:-1]
        
    if not os.path.exists(ipk_path):
        print(f"Error: Could not find file at {ipk_path}")
        return
        
    if not ipk_path.endswith(".ipk"):
        print("Error: File must end with .ipk")
        return
        
    filename = os.path.basename(ipk_path)
    print(f"\\nReading {filename} ({os.path.getsize(ipk_path)} bytes)...")
    
    with open(ipk_path, "rb") as f:
        ipk_data = f.read()
        
    print("Sending IPK to Modal Alpine container for lightning-fast APK conversion...")
    try:
        apk_name, apk_data = convert_ipk_to_apk.remote(ipk_data, filename)
        
        pkgs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Packages")
        os.makedirs(pkgs_dir, exist_ok=True)
        
        save_path = os.path.join(pkgs_dir, apk_name)
        with open(save_path, "wb") as f:
            f.write(apk_data)
            
        print(f"\\n✅ Success! Converted package to new APK format.")
        print(f"💾 Saved: {save_path}")
        print("It is now sitting in your Packages folder ready for the Firmware Builder!")
    except Exception as e:
        print(f"\\n❌ An error occurred during IPK conversion:\\n{e}")


def run_package_compiler():
    git_url = input("\nEnter the Git repository URL of the package to compile:\n> ").strip()
    if not git_url:
        print("Error: Git URL cannot be empty.")
        return
        
    print(f"\nSending compilation job to Modal (16 Cores / 16GB RAM)...")
    try:
        results = compile_package.remote(git_url)
        if not results:
            print("No APKs returned.")
            return
            
        print(f"\n✅ Compilation successful! Found {len(results)} .apk file(s).")
        
        # Save them into the Packages folder so they are ready for the firmware builder
        pkgs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Packages")
        os.makedirs(pkgs_dir, exist_ok=True)
        
        for filename, data in results.items():
            filepath = os.path.join(pkgs_dir, filename)
            with open(filepath, "wb") as f:
                f.write(data)
            print(f"💾 Saved: {filepath}")
            
    except Exception as e:
        print(f"\n❌ An error occurred during cloud compilation:\n{e}")


def run_firmware_builder():
    local_pkgs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Packages")
    os.makedirs(local_pkgs_dir, exist_ok=True)
    
    print(f"\nChecking for custom packages in: {local_pkgs_dir}")
    custom_apks = glob.glob(os.path.join(local_pkgs_dir, "*.apk"))
    if custom_apks:
        print(f"Found {len(custom_apks)} custom package(s) to inject.")
    else:
        print("No custom packages found. It will download everything from OpenWRT.")
        
    print("\nSending Firmware Compilation Job to Modal (16 Cores / 16GB RAM)...")
    try:
        results = build_firmware.remote()
        if not results:
            print("No firmware files returned.")
            return
            
        print(f"\n✅ Firmware build successful! Found {len(results)} image file(s). Downloading...")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for filename, data in results.items():
            filepath = os.path.join(script_dir, filename)
            with open(filepath, "wb") as f:
                f.write(data)
            print(f"💾 Saved: {filepath}")
            
    except Exception as e:
        print(f"\n❌ An error occurred during cloud firmware build:\n{e}")


@app.local_entrypoint()
def main():
    while True:
        print("\n==================================================")
        print(" 🚀 OpenWRT Ultimate Toolkit (Modal) 🚀")
        print("==================================================")
        print("Target: Netgear R6900v2 (ramips/mt7621) - Snapshot (APK)")
        print("--------------------------------------------------")
        print("1. Compile a custom OpenWRT package (.apk) from GitHub")
        print("2. Generate custom OpenWRT firmware (.bin)")
        print("3. Convert an old .ipk package to the new .apk format")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            run_package_compiler()
        elif choice == '2':
            run_firmware_builder()
        elif choice == '3':
            run_ipk_converter()
            break
        elif choice == '4':
            print("Exiting toolkit...")
            break
        else:
            print("Invalid choice, please select 1, 2, 3, or 4.")
