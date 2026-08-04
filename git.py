import subprocess
from datetime import datetime

# Stage all files
subprocess.run(["git", "add", "."], check=True)

# Commit
commit_message = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# commit_message = input("Enter your commit:\n")

commit_result = subprocess.run(
    ["git", "commit", "-m", commit_message],
    capture_output=True,
    text=True
)

print(commit_result.stdout)
print(commit_result.stderr)

# Push
push_result = subprocess.run(
    ["git", "push"],
    capture_output=True,
    text=True
)

print(push_result.stdout)
print(push_result.stderr)