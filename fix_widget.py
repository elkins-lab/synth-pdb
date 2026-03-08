import subprocess

try:
    subprocess.run(["jupyter", "nbextension", "enable", "--py", "--sys-prefix", "py3Dmol"])
    print("Widget enabled.")
except:
    pass
