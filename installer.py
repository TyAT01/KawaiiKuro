import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import sys
import os
import urllib.request
import zipfile
import threading
import io

class KuroInstaller(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kuro Installer")
        self.geometry("400x300")

        self.install_path = tk.StringVar()
        self.install_path.set(os.getcwd())

        self.create_widgets()

    def create_widgets(self):
        # Installation path
        tk.Label(self, text="Install Location:").pack(pady=5)
        path_frame = tk.Frame(self)
        path_frame.pack(fill=tk.X, padx=10)
        tk.Entry(path_frame, textvariable=self.install_path, state="readonly").pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(path_frame, text="Browse...", command=self.browse_directory).pack(side=tk.RIGHT)

        # Install button
        self.install_button = tk.Button(self, text="Install Kuro", command=self.start_installation)
        self.install_button.pack(pady=20)

        # Status console
        self.status_console = tk.Text(self, height=10, state="disabled", wrap="word", bg="black", fg="white")
        self.status_console.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    def browse_directory(self):
        path = filedialog.askdirectory(initialdir=self.install_path.get())
        if path:
            self.install_path.set(path)

    def log(self, message):
        self.status_console.config(state="normal")
        self.status_console.insert(tk.END, message + "\n")
        self.status_console.config(state="disabled")
        self.status_console.see(tk.END)
        self.update_idletasks()

    def start_installation(self):
        self.install_button.config(state="disabled")
        self.log("Starting installation...")
        # Run installation in a separate thread to keep the GUI responsive
        threading.Thread(target=self.install_kuro, daemon=True).start()

    def install_kuro(self):
        repo_url = "https://github.com/TyAT01/KawaiiKuro/archive/refs/heads/main.zip"
        install_path = self.install_path.get()

        if not os.path.isdir(install_path):
            self.log(f"Error: Installation path '{install_path}' does not exist.")
            messagebox.showerror("Error", f"Installation path does not exist.")
            self.install_button.config(state="normal")
            return

        try:
            # 1. Download the repository
            self.log("Downloading Kuro from GitHub...")
            with urllib.request.urlopen(repo_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download repository. Status code: {response.status}")
                zip_file = io.BytesIO(response.read())
            self.log("Download complete.")

            # 2. Extract the repository
            self.log("Extracting files...")
            with zipfile.ZipFile(zip_file) as zf:
                # The files are usually inside a directory like 'repo-main'.
                # We need to move them to the selected install_path.
                root_folder = zf.namelist()[0]
                for member in zf.infolist():
                    # Extract to install_path, stripping the root folder
                    if member.filename.startswith(root_folder):
                        target_path = os.path.join(install_path, member.filename[len(root_folder):])
                        if member.is_dir():
                            if not os.path.exists(target_path):
                                os.makedirs(target_path)
                        else:
                            with open(target_path, 'wb') as f:
                                f.write(zf.read(member.filename))

            self.log("Extraction complete.")

            # 3. Install dependencies
            self.log("Installing dependencies...")
            requirements_path = os.path.join(install_path, "requirements.txt")
            if not os.path.exists(requirements_path):
                self.log("Error: requirements.txt not found in the repository.")
                messagebox.showerror("Error", "Could not find requirements.txt.")
                self.install_button.config(state="normal")
                return

            # Use sys.executable to ensure pip is from the correct python installation
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
            self.log("Dependencies installed successfully.")

            self.log("\nInstallation finished!")
            self.log(f"You can now run Kuro from the installation directory.")
            messagebox.showinfo("Success", "Kuro has been installed successfully!")

        except zipfile.BadZipFile as e:
            self.log(f"Error extracting repository: {e}")
            messagebox.showerror("Error", f"Failed to extract repository:\n{e}")
        except subprocess.CalledProcessError as e:
            self.log(f"Error installing dependencies: {e}")
            messagebox.showerror("Error", f"Failed to install dependencies:\n{e}")
        except Exception as e:
            self.log(f"An unexpected error occurred: {e}")
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")
        finally:
            self.install_button.config(state="normal")


if __name__ == "__main__":
    app = KuroInstaller()
    app.mainloop()
