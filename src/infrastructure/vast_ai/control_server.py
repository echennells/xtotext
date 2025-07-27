"""
Control Server for managing transcription pipeline
Handles: downloading, audio extraction, file transfers, job coordination
"""
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid
import shutil

from .ssh_connection import SSHConnection
from .config import REMOTE_WORKSPACE


class ControlServer:
    """Control server for managing the transcription pipeline"""
    
    def __init__(self, ssh_connection: SSHConnection):
        self.ssh = ssh_connection
        self.workspace = f"{REMOTE_WORKSPACE}/control"
        self.audio_dir = f"{self.workspace}/audio"
        self.queue_file = f"{self.workspace}/job_queue.json"
        self.setup_directories()
        
    def setup_directories(self):
        """Set up remote directories"""
        print("Setting up control server directories...")
        self.ssh.execute_command(
            f"mkdir -p {self.workspace} {self.audio_dir}"
        )
        
        
    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Extract audio from video file using ffmpeg
        
        Args:
            video_path: Path to video file
            output_path: Optional output path
            
        Returns:
            Path to extracted audio
        """
        if not output_path:
            base_name = Path(video_path).stem
            output_path = f"{self.audio_dir}/{base_name}.mp3"
            
        print(f"Extracting audio from {video_path}...")
        
        cmd = (
            f"ffmpeg -i '{video_path}' -vn -acodec mp3 "
            f"-ab 192k -ar 44100 -y '{output_path}'"
        )
        
        ret, out, err = self.ssh.execute_command(cmd, timeout=300)
        
        if ret != 0:
            print(f"Audio extraction failed: {err}")
            return None
            
        print(f"Audio extracted to: {output_path}")
        return output_path
        
    def transfer_to_gpu(self, file_path: str, gpu_ssh: SSHConnection, remote_path: str) -> bool:
        """
        Transfer file from control server to GPU instance
        
        Args:
            file_path: File on control server
            gpu_ssh: SSH connection to GPU instance
            remote_path: Destination path on GPU instance
            
        Returns:
            Success boolean
        """
        print(f"Transferring {file_path} to GPU instance...")
        
        # Create a transfer script that uses scp between instances
        transfer_script = f'''#!/bin/bash
# Transfer file from control to GPU instance

# First, ensure SSH key is available
if [ ! -f ~/.ssh/id_rsa ]; then
    ssh-keygen -t rsa -b 2048 -f ~/.ssh/id_rsa -N ""
fi

# Get the public key
pubkey=$(cat ~/.ssh/id_rsa.pub)

# Add key to GPU instance (this would need to be done via API or manually)
echo "Public key that needs to be added to GPU instance:"
echo "$pubkey"

# For now, use a direct download/upload approach
# Download to local temp
temp_file="/tmp/transfer_$(basename '{file_path}')"
cp "{file_path}" "$temp_file"

# The GPU instance would need to fetch this file
echo "File ready at: $temp_file"
'''
        
        script_path = f"{self.workspace}/transfer.sh"
        self.ssh.create_remote_script(transfer_script, script_path)
        self.ssh.execute_command(f"chmod +x {script_path}")
        
        # For now, we'll use a simpler approach - have the GPU instance pull the file
        # This requires the control server to serve the file (e.g., via HTTP)
        print("Transfer method needs to be implemented based on network setup")
        return False
        
    def create_job_queue(self, audio_files: List[str], model: str = "base") -> str:
        """
        Create a job queue for batch processing
        
        Args:
            audio_files: List of audio file paths
            model: Whisper model to use
            
        Returns:
            Queue ID
        """
        queue_id = str(uuid.uuid4())[:8]
        
        jobs = []
        for audio_file in audio_files:
            job = {
                "id": str(uuid.uuid4())[:8],
                "audio_file": audio_file,
                "model": model,
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
            jobs.append(job)
            
        queue = {
            "queue_id": queue_id,
            "created_at": datetime.now().isoformat(),
            "total_jobs": len(jobs),
            "completed": 0,
            "jobs": jobs
        }
        
        # Save queue to file
        queue_file = f"{self.workspace}/queue_{queue_id}.json"
        queue_json = json.dumps(queue, indent=2)
        
        # Create queue file on server
        self.ssh.execute_command(
            f"echo '{queue_json}' > {queue_file}"
        )
        
        print(f"Created job queue {queue_id} with {len(jobs)} jobs")
        return queue_id
        
    def get_queue_status(self, queue_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a job queue"""
        queue_file = f"{self.workspace}/queue_{queue_id}.json"
        
        ret, out, err = self.ssh.execute_command(f"cat {queue_file} 2>/dev/null")
        
        if ret != 0:
            return None
            
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return None
            
    def list_audio_files(self) -> List[Tuple[str, int]]:
        """
        List audio files on control server
        
        Returns:
            List of (filename, size_in_mb) tuples
        """
        cmd = f"find {self.audio_dir} -name '*.mp3' -type f -exec ls -l {{}} \\; | awk '{{print $9, $5/1024/1024}}'"
        
        ret, out, err = self.ssh.execute_command(cmd)
        
        if ret != 0:
            return []
            
        files = []
        for line in out.strip().split('\n'):
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    filepath = parts[0]
                    size_mb = float(parts[1])
                    files.append((filepath, size_mb))
                    
        return files
        
    def cleanup_old_files(self, days: int = 7):
        """Clean up files older than specified days"""
        print(f"Cleaning up files older than {days} days...")
        
        cmd = f"find {self.workspace} -type f -mtime +{days} -name '*.mp3' -delete"
        self.ssh.execute_command(cmd)
        
        cmd = f"find {self.workspace} -type f -mtime +{days} -name '*.mp4' -delete"
        self.ssh.execute_command(cmd)
        
        print("Cleanup complete")
        
    def get_server_stats(self) -> Dict[str, Any]:
        """Get control server statistics"""
        stats = {}
        
        # Disk usage
        ret, out, err = self.ssh.execute_command(f"df -h {self.workspace} | tail -1")
        if ret == 0:
            parts = out.strip().split()
            if len(parts) >= 4:
                stats['disk_used'] = parts[2]
                stats['disk_available'] = parts[3]
                stats['disk_percent'] = parts[4]
                
        # File counts
        ret, out, err = self.ssh.execute_command(
            f"find {self.audio_dir} -name '*.mp3' | wc -l"
        )
        if ret == 0:
            stats['audio_files'] = int(out.strip())
            
        # Total size
        ret, out, err = self.ssh.execute_command(
            f"du -sh {self.workspace} | cut -f1"
        )
        if ret == 0:
            stats['total_size'] = out.strip()
            
        return stats