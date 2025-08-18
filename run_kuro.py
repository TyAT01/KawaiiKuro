import os
import subprocess
import sys

def main():
    """Main function to run the setup and start the application."""
    install_flag = ".install_complete"

    if not os.path.exists(install_flag):
        print("First-time setup: Installing dependencies...")
        try:
            subprocess.check_call([sys.executable, "install_dependencies.py"])
            print("Setup complete.")
        except subprocess.CalledProcessError as e:
            print(f"Error during setup: {e}")
            print("Please try running 'python install_dependencies.py' manually.")
            sys.exit(1)
    else:
        print("Dependencies are already installed.")

    print("Starting Kuro...")
    try:
        subprocess.run([sys.executable, "kuro_app.py"] + sys.argv[1:])
    except Exception as e:
        print(f"Error starting Kuro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
