#!/usr/bin/env python3
"""
Check for running Digital Ocean droplets
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from infrastructure.digital_ocean.client import DigitalOceanClient

def main():
    print("Checking for running Digital Ocean droplets...")
    print("=" * 80)
    
    try:
        # Create client
        client = DigitalOceanClient()
        
        # List all droplets
        droplets = client.list_droplets()
        
        if not droplets:
            print("No droplets found.")
            return
        
        print(f"Found {len(droplets)} droplet(s):\n")
        
        # Check for xtotext droplets specifically
        xtotext_droplets = []
        
        for droplet in droplets:
            # Get IP address
            ip = None
            for network in droplet.get("networks", {}).get("v4", []):
                if network["type"] == "public":
                    ip = network["ip_address"]
                    break
            
            # Check if it's an xtotext droplet
            is_xtotext = any(tag in droplet.get("tags", []) for tag in ["xtotext", "transcription", "temp"])
            if is_xtotext:
                xtotext_droplets.append(droplet)
            
            # Print droplet info
            print(f"Droplet: {droplet['name']}")
            print(f"  ID: {droplet['id']}")
            print(f"  Status: {droplet['status']}")
            print(f"  IP: {ip or 'No public IP'}")
            print(f"  Region: {droplet['region']['name']}")
            print(f"  Size: {droplet['size']['slug']}")
            print(f"  Created: {droplet['created_at']}")
            print(f"  Tags: {', '.join(droplet.get('tags', []))}")
            print(f"  Is xtotext: {'Yes' if is_xtotext else 'No'}")
            print()
        
        # Special section for xtotext droplets
        if xtotext_droplets:
            print("=" * 80)
            print("XTOTEXT DROPLETS:")
            print("=" * 80)
            
            for droplet in xtotext_droplets:
                # Get IP
                ip = None
                for network in droplet.get("networks", {}).get("v4", []):
                    if network["type"] == "public":
                        ip = network["ip_address"]
                        break
                
                print(f"\nDroplet: {droplet['name']}")
                print(f"  SSH: ssh root@{ip}")
                print(f"  ID: {droplet['id']}")
                print(f"  Status: {droplet['status']}")
                print(f"  Created: {droplet['created_at']}")
                
                # Save to file for easy access
                with open("current_droplet.txt", "w") as f:
                    f.write(f"IP: {ip}\n")
                    f.write(f"ID: {droplet['id']}\n")
                    f.write(f"Name: {droplet['name']}\n")
                    f.write(f"SSH: ssh root@{ip}\n")
                
                print(f"\n  Droplet info saved to: current_droplet.txt")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()