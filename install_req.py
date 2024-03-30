import subprocess
import sys

requirements_file = 'requirements.txt'

with open(requirements_file, 'r') as file:
    for line in file:
        package = line.strip()
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}. Reason: {str(e)}")

