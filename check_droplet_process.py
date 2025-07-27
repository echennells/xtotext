#!/usr/bin/env python3
"""
Check what the prediction processing script is doing on the droplet
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
    
    # Check Python process details
    print("Python process details:")
    print("-" * 40)
    exit_code, stdout, stderr = ssh.execute_command("ps auxww | grep 'python.*process_predictions' | grep -v grep")
    if stdout:
        print(stdout)
        # Extract PID
        pid = stdout.strip().split()[1]
        print(f"\nProcess ID: {pid}")
        
        # Check CPU and memory usage
        print("\nProcess resource usage:")
        print("-" * 40)
        exit_code, stdout, stderr = ssh.execute_command(f"top -b -n 1 -p {pid} | tail -2")
        print(stdout)
        
        # Check open files
        print("\nOpen files by the process:")
        print("-" * 40)
        exit_code, stdout, stderr = ssh.execute_command(f"lsof -p {pid} 2>/dev/null | grep -E '(REG|DIR)' | tail -20")
        if stdout:
            print(stdout)
        
        # Check if process is making network connections
        print("\nNetwork connections:")
        print("-" * 40)
        exit_code, stdout, stderr = ssh.execute_command(f"lsof -p {pid} -i 2>/dev/null")
        if stdout:
            print(stdout)
        else:
            print("No network connections")
            
        # Check strace (last 50 system calls)
        print("\nLast system calls (strace for 2 seconds):")
        print("-" * 40)
        exit_code, stdout, stderr = ssh.execute_command(f"timeout 2 strace -p {pid} 2>&1 | tail -20")
        if stdout:
            print(stdout)
        elif stderr:
            print(stderr)
    
    # Check the content of the prediction script
    print("\nPrediction script first 50 lines:")
    print("-" * 40)
    exit_code, stdout, stderr = ssh.execute_command("head -50 /workspace/xtotext/scripts/digital_ocean/process_predictions.py")
    print(stdout)
    
    # Check if there's any output being generated
    print("\nChecking for recent file modifications:")
    print("-" * 40)
    exit_code, stdout, stderr = ssh.execute_command("find /workspace -type f -mmin -5 -ls 2>/dev/null")
    if stdout:
        print(stdout)
    else:
        print("No files modified in the last 5 minutes")
    
    # Check disk I/O
    print("\nDisk I/O statistics:")
    print("-" * 40)
    exit_code, stdout, stderr = ssh.execute_command("iostat -x 1 2 | tail -20")
    if exit_code == 0:
        print(stdout)
    else:
        # Try without iostat
        exit_code, stdout, stderr = ssh.execute_command("cat /proc/diskstats | grep vda")
        print(stdout)
    
    ssh.close()
    print("\n" + "=" * 80)
    print(f"SSH command to connect: ssh root@{ip}")
    print(f"To check the process: ssh root@{ip} 'ps aux | grep process_predictions'")
    print(f"To tail any logs: ssh root@{ip} 'tail -f /workspace/*.log'")

if __name__ == "__main__":
    main()