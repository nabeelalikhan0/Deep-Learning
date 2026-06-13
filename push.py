import subprocess
from datetime import datetime

subprocess.run(["git","add","."])
commit_result = subprocess.run(["git","commit","-m",f'"{datetime.now()}"'],shell=True,capture_output=True)

print(commit_result.stdout)