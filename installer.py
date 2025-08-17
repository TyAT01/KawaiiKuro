import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import urllib.request
import zipfile
import threading
import io
import datetime
import shutil
import json
import hashlib
import re

class KuroInstaller(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kuro Installer")
        self.geometry("420x360")
        self.resizable(True, True)

        self.install_path = tk.StringVar()
        self.install_path.set(os.getcwd())

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self, text="Install Location:").pack(pady=5)
        path_frame = tk.Frame(self)
        path_frame.pack(fill=tk.X, padx=10)
        tk.Entry(path_frame, textvariable=self.install_path, state="readonly").pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(path_frame, text="Browse...", command=self.browse_directory).pack(side=tk.RIGHT)

        self.install_button = tk.Button(self, text="Install Kuro", command=self.start_installation)
        self.install_button.pack(pady=20)

        console_frame = tk.Frame(self)
        console_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(console_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_console = tk.Text(console_frame, height=12, state="disabled", wrap="word", bg="black", fg="white", yscrollcommand=scrollbar.set)
        self.status_console.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.status_console.yview)

    def browse_directory(self):
        path = filedialog.askdirectory(initialdir=self.install_path.get())
        if path:
            try:
                test_file = os.path.join(path, "test_write.txt")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                self.install_path.set(path)
            except PermissionError:
                messagebox.showerror("Error", "Selected directory is not writable. Please choose another.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to access directory: {e}")

    def log(self, message):
        self.status_console.config(state="normal")
        self.status_console.insert(tk.END, message + "\n")
        self.status_console.config(state="disabled")
        self.status_console.see(tk.END)
        self.update_idletasks()

    def start_installation(self):
        self.install_button.config(state="disabled")
        self.log("Starting installation...")
        threading.Thread(target=self.install_kuro, daemon=True).start()

    def get_latest_python_311(self):
        """Fetch the latest Python 3.11.x embeddable zip URL and checksum."""
        base_url = "https://www.python.org/ftp/python/"
        try:
            with urllib.request.urlopen(base_url, timeout=30) as response:
                html = response.read().decode()
            versions = re.findall(r'href="(\d+\.\d+\.\d+)/"', html)
            python_311_versions = [v for v in versions if v.startswith("3.11.")]
            if not python_311_versions:
                raise Exception("No Python 3.11 versions found")
            latest_version = max(python_311_versions, key=lambda v: [int(x) for x in v.split(".")])
            zip_url = f"{base_url}{latest_version}/python-{latest_version}-embed-amd64.zip"
            # Fetch checksum (assuming Python.org provides a checksums file; adjust if needed)
            checksum_url = f"{base_url}{latest_version}/python-{latest_version}-embed-amd64.zip.sha256"
            try:
                with urllib.request.urlopen(checksum_url, timeout=30) as response:
                    expected_sha256 = response.read().decode().strip().split()[0]
            except:
                expected_sha256 = None  # Fallback if checksum not available
            return zip_url, latest_version, expected_sha256
        except Exception as e:
            raise Exception(f"Failed to find latest Python 3.11: {e}")

    def ensure_python(self, install_path):
        python_embed_dir = os.path.join(install_path, "python_embed")
        python_exe = os.path.join(python_embed_dir, "python.exe")
        version_file = os.path.join(install_path, "python_version.txt")

        if os.path.exists(python_exe):
            self.log("Found portable Python.")
            return python_exe, None

        self.log("Fetching latest Python 3.11...")
        try:
            zip_url, version, expected_sha256 = self.get_latest_python_311()
            self.log(f"Downloading Python {version}...")
            with urllib.request.urlopen(zip_url, timeout=30) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download Python. Status code: {response.status}")
                zip_file = io.BytesIO(response.read())

            # Verify checksum
            if expected_sha256:
                sha256_hash = hashlib.sha256(zip_file.getvalue()).hexdigest()
                if sha256_hash != expected_sha256:
                    raise ValueError("Python zip checksum mismatch")
                self.log("Python zip checksum verified.")
            else:
                self.log("Warning: No checksum available for Python zip.")

            self.log("Extracting portable Python...")
            with zipfile.ZipFile(zip_file) as zf:
                for member in zf.infolist():
                    target_path = os.path.join(python_embed_dir, member.filename)
                    if os.path.commonpath([python_embed_dir, os.path.abspath(target_path)]) != os.path.abspath(python_embed_dir):
                        raise ValueError(f"Zip slip attempt detected: {member.filename}")
                    if member.is_dir():
                        os.makedirs(target_path, exist_ok=True)
                    else:
                        with open(target_path, 'wb') as f:
                            f.write(zf.read(member.filename))

            if not os.path.exists(python_exe):
                raise FileNotFoundError("Python executable not found after extraction")
            
            # Save Python version
            with open(version_file, "w") as f:
                f.write(version)
            self.log(f"Portable Python {version} installed.")
            return python_exe, version

        except Exception as e:
            self.log(f"Error downloading/extracting Python: {e}")
            if os.path.exists(python_embed_dir):
                shutil.rmtree(python_embed_dir, ignore_errors=True)
            raise

    def install_kuro(self):
        repo_url = "https://github.com/TyAT01/KawaiiKuro/archive/refs/heads/main.zip"
        install_path = self.install_path.get()
        venv_path = os.path.join(install_path, "kuro_env")

        if not os.path.isdir(install_path):
            self.log(f"Error: Installation path '{install_path}' does not exist.")
            messagebox.showerror("Error", "Installation path does not exist.")
            self.install_button.config(state="normal")
            return

        try:
            # Ensure Python exists
            python_exe, python_version = self.ensure_python(install_path)

            # Download repo
            self.log("Downloading Kuro from GitHub...")
            try:
                with urllib.request.urlopen(repo_url, timeout=30) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download repository. Status code: {response.status}")
                    zip_file = io.BytesIO(response.read())
            except urllib.error.URLError as e:
                raise Exception(f"Network error downloading repository: {e}")

            # Extract repo
            self.log("Extracting files...")
            with zipfile.ZipFile(zip_file) as zf:
                root_folder = zf.namelist()[0]
                for member in zf.infolist():
                    if member.filename.startswith(root_folder):
                        target_path = os.path.join(install_path, member.filename[len(root_folder):])
                        if os.path.commonpath([install_path, os.path.abspath(target_path)]) != os.path.abspath(install_path):
                            raise ValueError(f"Zip slip attempt detected: {member.filename}")
                        if member.is_dir():
                            os.makedirs(target_path, exist_ok=True)
                        else:
                            with open(target_path, 'wb') as f:
                                f.write(zf.read(member.filename))
            self.log("Extraction complete.")

            # Validate main.py
            main_py = os.path.join(install_path, "main.py")
            if not os.path.exists(main_py):
                raise FileNotFoundError("main.py not found in repository")

            # Create venv
            self.log("Creating virtual environment...")
            if not os.path.exists(venv_path):
                subprocess.check_call([python_exe, "-m", "venv", venv_path])
            self.log("Virtual environment ready at " + venv_path)

            pip_executable = os.path.join(venv_path, "Scripts", "pip.exe")

            # Install dependencies
            self.log("Installing dependencies into venv...")
            requirements_path = os.path.join(install_path, "requirements.txt")
            if not os.path.exists(requirements_path):
                raise FileNotFoundError("requirements.txt not found")
            subprocess.check_call([pip_executable, "install", "-r", requirements_path])
            self.log("Dependencies installed successfully.")

            # Create version.txt
            version_file = os.path.join(install_path, "version.txt")
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(version_file, "w") as f:
                f.write(f"Kuro Version 1.0.0 - Installed {now}\n")

            # Create run script
            run_script = os.path.join(install_path, "run_kuro.bat")
            with open(run_script, "w") as f:
                f.write(f"""@echo off
:: Auto-update check (runs only if online)
call "%~dp0\\kuro_env\\Scripts\\activate"
python "%~dp0\\auto_update.py"

:: Show version if available
if exist "%~dp0\\version.txt" (
    echo -------------------------------
    type "%~dp0\\version.txt"
    echo -------------------------------
)

:: Run Kuro
python "%~dp0\\main.py"
pause
""")

            # Create uninstaller with confirmation
            uninstaller = os.path.join(install_path, "uninstall_kuro.bat")
            with open(uninstaller, "w") as f:
                f.write(f"""@echo off
echo WARNING: This will delete all Kuro files.
set /p confirm=Are you sure you want to uninstall Kuro? (y/N): 
if /i not "%confirm%"=="y" (
    echo Uninstallation cancelled.
    pause
    exit /b
)
echo Uninstalling Kuro...
rmdir /s /q "%~dp0\\kuro_env"
rmdir /s /q "%~dp0\\python_embed"
del /q "%~dp0\\*"
echo Kuro has been uninstalled.
pause
""")

            # Create manual updater
            updater = os.path.join(install_path, "update_kuro.py")
            with open(updater, "w") as f:
                f.write(self.make_update_script(repo_url, install_path, venv_path))

            # Create auto-updater
            auto_update = os.path.join(install_path, "auto_update.py")
            with open(auto_update, "w") as f:
                f.write(self.make_auto_update_script(repo_url))

            self.log("\nInstallation finished!")
            self.log(f"- Run Kuro: run_kuro.bat\n- Update manually: update_kuro.py\n- Uninstall: uninstall_kuro.bat")
            messagebox.showinfo("Success", "Kuro installed!\nUse run_kuro.bat to run.\nupdate_kuro.py to update.\nuninstall_kuro.bat to remove.")

        except Exception as e:
            self.log(f"An error occurred: {e}")
            messagebox.showerror("Error", f"An error occurred:\n{e}")
            for dir in ["python_embed", "kuro_env"]:
                dir_path = os.path.join(install_path, dir)
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path, ignore_errors=True)
        finally:
            self.install_button.config(state="normal")

    def make_update_script(self, repo_url, install_path, venv_path):
        return f'''import urllib.request, zipfile, io, os, subprocess, hashlib, datetime, json, re

repo_url = "{repo_url}"
install_path = r"{install_path}"
venv_path = r"{venv_path}"
pip_executable = os.path.join(venv_path, "Scripts", "pip.exe")
hash_file = os.path.join(install_path, "repo_hash.json")
python_embed_dir = os.path.join(install_path, "python_embed")
python_exe = os.path.join(python_embed_dir, "python.exe")
python_version_file = os.path.join(install_path, "python_version.txt")

def get_hash_from_zip(zip_bytes):
    hasher = hashlib.sha256()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for f in zf.infolist():
            if not f.is_dir():
                hasher.update(zf.read(f.filename))
    return hasher.hexdigest()

def get_latest_python_311():
    base_url = "https://www.python.org/ftp/python/"
    try:
        with urllib.request.urlopen(base_url, timeout=30) as response:
            html = response.read().decode()
        versions = re.findall(r'href="(\d+\.\d+\.\d+)/"', html)
        python_311_versions = [v for v in versions if v.startswith("3.11.")]
        if not python_311_versions:
            raise Exception("No Python 3.11 versions found")
        latest_version = max(python_311_versions, key=lambda v: [int(x) for x in v.split(".")])
        zip_url = f"{{base_url}}{{latest_version}}/python-{{latest_version}}-embed-amd64.zip"
        checksum_url = f"{{base_url}}{{latest_version}}/python-{{latest_version}}-embed-amd64.zip.sha256"
        try:
            with urllib.request.urlopen(checksum_url, timeout=30) as response:
                expected_sha256 = response.read().decode().strip().split()[0]
        except:
            expected_sha256 = None
        return zip_url, latest_version, expected_sha256
    except Exception as e:
        raise Exception(f"Failed to find latest Python 3.11: {{e}}")

# Check for Python update
print("Checking for Python update...")
try:
    zip_url, latest_version, expected_sha256 = get_latest_python_311()
    current_version = None
    if os.path.exists(python_version_file):
        with open(python_version_file, "r") as f:
            current_version = f.read().strip()
    
    if latest_version != current_version:
        print(f"New Python version {{latest_version}} found! Updating...")
        with urllib.request.urlopen(zip_url, timeout=30) as response:
            if response.status != 200:
                raise Exception(f"Failed to download Python. Status: {{response.status}}")
            zip_file = io.BytesIO(response.read())
        
        if expected_sha256:
            sha256_hash = hashlib.sha256(zip_file.getvalue()).hexdigest()
            if sha256_hash != expected_sha256:
                raise ValueError("Python zip checksum mismatch")
            print("Python zip checksum verified.")
        
        # Backup existing Python
        if os.path.exists(python_embed_dir):
            shutil.rmtree(python_embed_dir + "_backup", ignore_errors=True)
            shutil.move(python_embed_dir, python_embed_dir + "_backup")
        
        print("Extracting new Python...")
        with zipfile.ZipFile(zip_file) as zf:
            for member in zf.infolist():
                target_path = os.path.join(python_embed_dir, member.filename)
                if os.path.commonpath([python_embed_dir, os.path.abspath(target_path)]) != os.path.abspath(python_embed_dir):
                    raise ValueError(f"Zip slip attempt detected: {{member.filename}}")
                if member.is_dir():
                    os.makedirs(target_path, exist_ok=True)
                else:
                    with open(target_path, 'wb') as f:
                        f.write(zf.read(member.filename))
        
        if not os.path.exists(python_exe):
            raise FileNotFoundError("Python executable not found after update")
        
        with open(python_version_file, "w") as f:
            f.write(latest_version)
        
        # Recreate venv
        print("Recreating virtual environment...")
        shutil.rmtree(venv_path, ignore_errors=True)
        subprocess.check_call([python_exe, "-m", "venv", venv_path])
        print("Python updated to " + latest_version)
    else:
        print("Python is up to date.")
except Exception as e:
    print(f"Failed to update Python: {{e}}. Using existing version.")

# Check for Kuro update
print("Checking for Kuro update...")
try:
    with urllib.request.urlopen(repo_url, timeout=30) as response:
        if response.status != 200:
            raise Exception("Failed to download repository")
        zip_bytes = response.read()

    latest_hash = get_hash_from_zip(zip_bytes)
    old_hash = None
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            old_hash = json.load(f).get("hash")

    if latest_hash != old_hash:
        print("New Kuro version found! Updating...")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            root_folder = zf.namelist()[0]
            for member in zf.infolist():
                if member.filename.startswith(root_folder):
                    target_path = os.path.join(install_path, member.filename[len(root_folder):])
                    if os.path.commonpath([install_path, os.path.abspath(target_path)]) != os.path.abspath(install_path):
                        raise ValueError(f"Zip slip attempt detected: {{member.filename}}")
                    if member.is_dir():
                        os.makedirs(target_path, exist_ok=True)
                    else:
                        with open(target_path, 'wb') as f:
                            f.write(zf.read(member.filename))
        with open(hash_file, "w") as f:
            json.dump({{"hash": latest_hash}}, f)

        requirements_path = os.path.join(install_path, "requirements.txt")
        if os.path.exists(requirements_path):
            subprocess.call([pip_executable, "install", "-r", requirements_path])

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(install_path, "version.txt"), "w") as f:
            f.write(f"Kuro updated on {{now}} - {{latest_hash[:8]}}\\n")

        print("Kuro update complete.")
    else:
        print("Kuro is already up to date.")
except Exception as e:
    print(f"Failed to update Kuro: {{e}}. Running locally.")
'''

    def make_auto_update_script(self, repo_url):
        return f'''import urllib.request, zipfile, io, os, subprocess, hashlib, json, datetime, re, shutil

repo_url = "{repo_url}"
install_path = os.path.dirname(os.path.abspath(__file__))
venv_path = os.path.join(install_path, "kuro_env")
pip_executable = os.path.join(venv_path, "Scripts", "pip.exe")
hash_file = os.path.join(install_path, "repo_hash.json")
python_embed_dir = os.path.join(install_path, "python_embed")
python_exe = os.path.join(python_embed_dir, "python.exe")
python_version_file = os.path.join(install_path, "python_version.txt")

def get_hash_from_zip(zip_bytes):
    hasher = hashlib.sha256()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for f in zf.infolist():
            if not f.is_dir():
                hasher.update(zf.read(f.filename))
    return hasher.hexdigest()

def get_latest_python_311():
    base_url = "https://www.python.org/ftp/python/"
    try:
        with urllib.request.urlopen(base_url, timeout=30) as response:
            html = response.read().decode()
        versions = re.findall(r'href="(\d+\.\d+\.\d+)/"', html)
        python_311_versions = [v for v in versions if v.startswith("3.11.")]
        if not python_311_versions:
            raise Exception("No Python 3.11 versions found")
        latest_version = max(python_311_versions, key=lambda v: [int(x) for x in v.split(".")])
        zip_url = f"{{base_url}}{{latest_version}}/python-{{latest_version}}-embed-amd64.zip"
        checksum_url = f"{{base_url}}{{latest_version}}/python-{{latest_version}}-embed-amd64.zip.sha256"
        try:
            with urllib.request.urlopen(checksum_url, timeout=30) as response:
                expected_sha256 = response.read().decode().strip().split()[0]
        except:
            expected_sha256 = None
        return zip_url, latest_version, expected_sha256
    except Exception as e:
        raise Exception(f"Failed to find latest Python 3.11: {{e}}")

try:
    # Check for Python update
    print("Checking for Python update...")
    zip_url, latest_version, expected_sha256 = get_latest_python_311()
    current_version = None
    if os.path.exists(python_version_file):
        with open(python_version_file, "r") as f:
            current_version = f.read().strip()
    
    if latest_version != current_version:
        print(f"New Python version {{latest_version}} found! Updating...")
        with urllib.request.urlopen(zip_url, timeout=30) as response:
            if response.status != 200:
                raise Exception(f"Failed to download Python. Status: {{response.status}}")
            zip_file = io.BytesIO(response.read())
        
        if expected_sha256:
            sha256_hash = hashlib.sha256(zip_file.getvalue()).hexdigest()
            if sha256_hash != expected_sha256:
                raise ValueError("Python zip checksum mismatch")
            print("Python zip checksum verified.")
        
        # Backup existing Python
        if os.path.exists(python_embed_dir):
            shutil.rmtree(python_embed_dir + "_backup", ignore_errors=True)
            shutil.move(python_embed_dir, python_embed_dir + "_backup")
        
        print("Extracting new Python...")
        with zipfile.ZipFile(zip_file) as zf:
            for member in zf.infolist():
                target_path = os.path.join(python_embed_dir, member.filename)
                if os.path.commonpath([python_embed_dir, os.path.abspath(target_path)]) != os.path.abspath(python_embed_dir):
                    raise ValueError(f"Zip slip attempt detected: {{member.filename}}")
                if member.is_dir():
                    os.makedirs(target_path, exist_ok=True)
                else:
                    with open(target_path, 'wb') as f:
                        f.write(zf.read(member.filename))
        
        if not os.path.exists(python_exe):
            raise FileNotFoundError("Python executable not found after update")
        
        with open(python_version_file, "w") as f:
            f.write(latest_version)
        
        # Recreate venv
        print("Recreating virtual environment...")
        shutil.rmtree(venv_path, ignore_errors=True)
        subprocess.check_call([python_exe, "-m", "venv", venv_path])
        print("Python updated to " + latest_version)

    # Check for Kuro update
    print("Checking for Kuro update...")
    with urllib.request.urlopen(repo_url, timeout=30) as response:
        if response.status != 200:
            raise Exception("Bad status")
        zip_bytes = response.read()

    latest_hash = get_hash_from_zip(zip_bytes)
    old_hash = None
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            old_hash = json.load(f).get("hash")

    if latest_hash != old_hash:
        print("New Kuro version found! Updating...")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            root_folder = zf.namelist()[0]
            for member in zf.infolist():
                if member.filename.startswith(root_folder):
                    target_path = os.path.join(install_path, member.filename[len(root_folder):])
                    if os.path.commonpath([install_path, os.path.abspath(target_path)]) != os.path.abspath(install_path):
                        raise ValueError(f"Zip slip attempt detected: {{member.filename}}")
                    if member.is_dir():
                        os.makedirs(target_path, exist_ok=True)
                    else:
                        with open(target_path, 'wb') as f:
                            f.write(zf.read(member.filename))
        with open(hash_file, "w") as f:
            json.dump({{"hash": latest_hash}}, f)

        requirements_path = os.path.join(install_path, "requirements.txt")
        if os.path.exists(requirements_path):
            subprocess.call([pip_executable, "install", "-r", requirements_path])

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(install_path, "version.txt"), "w") as f:
            f.write(f"Kuro updated on {{now}} - {{latest_hash[:8]}}\\n")

        print("Kuro updated successfully!")
except Exception:
    print("Offline or unable to check updates. Running Kuro locally.")
'''

if __name__ == "__main__":
    app = KuroInstaller()
    app.mainloop()
