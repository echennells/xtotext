#!/usr/bin/env python3
"""
Debug what's happening on the Digital Ocean droplet
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from infrastructure.digital_ocean.ssh_connection import DOSSHConnection

def main():
    # Read droplet info
    if not Path("current_droplet.txt").exists():
        print("Error: current_droplet.txt not found. Run check_droplets.py first.")
        return
    
    with open("current_droplet.txt", "r") as f:
        lines = f.readlines()
    
    ip = None
    for line in lines:
        if line.startswith("IP:"):
            ip = line.split(":")[1].strip()
            break
    
    if not ip:
        print("Error: Could not find IP in current_droplet.txt")
        return
    
    print(f"Connecting to droplet at {ip}...")
    print("=" * 80)
    
    # Connect via SSH
    ssh = DOSSHConnection(ip)
    if not ssh.connect():
        print("Failed to connect via SSH")
        return
    
    print("Connected successfully!\n")
    
    # Check running processes
    print("Checking running Python processes:")
    print("-" * 40)
    exit_code, stdout, stderr = ssh.execute_command("ps aux | grep python | grep -v grep")
    if stdout:
        print(stdout)
    else:
        print("No Python processes running")
    
    # Check if there's a venv
    print("\nChecking for virtual environment:")
    print("-" * 40)
    exit_code, stdout, stderr = ssh.execute_command("ls -la /workspace/venv/")
    if exit_code == 0:
        print("Virtual environment exists")
    else:
        print("No virtual environment found")
    
    # Check workspace contents
    print("\nWorkspace contents:")
    print("-" * 40)
    exit_code, stdout, stderr = ssh.execute_command("ls -la /workspace/")
    print(stdout)
    
    # Check xtotext directory
    print("\nXtotext directory contents:")
    print("-" * 40)
    exit_code, stdout, stderr = ssh.execute_command("ls -la /workspace/xtotext/")
    if exit_code == 0:
        print(stdout)
    else:
        print("Xtotext directory not found")
    
    # Check for log files
    print("\nChecking for log files:")
    print("-" * 40)
    exit_code, stdout, stderr = ssh.execute_command("find /workspace -name '*.log' -type f 2>/dev/null | head -20")
    if stdout:
        print(stdout)
    else:
        print("No log files found")
    
    # Check last few lines of cloud-init log
    print("\nLast 20 lines of cloud-init output log:")
    print("-" * 40)
    exit_code, stdout, stderr = ssh.execute_command("tail -20 /var/log/cloud-init-output.log")
    if exit_code == 0:
        print(stdout)
    
    # Check system resources
    print("\nSystem resources:")
    print("-" * 40)
    exit_code, stdout, stderr = ssh.execute_command("free -h")
    print(stdout)
    
    print("\nDisk usage:")
    print("-" * 40)
    exit_code, stdout, stderr = ssh.execute_command("df -h /")
    print(stdout)
    
    print("\nCPU info:")
    print("-" * 40)
    exit_code, stdout, stderr = ssh.execute_command("lscpu | grep -E 'Model name:|CPU\\(s\\):'")
    print(stdout)
    
    # Check for any error in prediction processing
    print("\nChecking for prediction processing errors:")
    print("-" * 40)
    exit_code, stdout, stderr = ssh.execute_command("grep -i error /workspace/*.log 2>/dev/null | tail -20")
    if stdout:
        print(stdout)
    else:
        print("No errors found in log files")
    
    # Check if prediction script exists
    print("\nChecking for prediction processing script:")
    print("-" * 40)
    exit_code, stdout, stderr = ssh.execute_command("ls -la /workspace/xtotext/scripts/digital_ocean/process_predictions.py")
    if exit_code == 0:
        print("Script exists:")
        print(stdout)
    else:
        print("Script not found!")
    
    ssh.close()
    print("\n" + "=" * 80)
    print(f"To connect manually: ssh root@{ip}")

if __name__ == "__main__":
    main()