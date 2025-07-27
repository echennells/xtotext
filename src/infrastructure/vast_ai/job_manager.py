"""
Job Manager for GPU Transcription Tasks
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid

from .ssh_connection import SSHConnection
from .config import REMOTE_WORKSPACE, REMOTE_AUDIO_DIR, REMOTE_OUTPUT_DIR


class TranscriptionJob:
    """Represents a transcription job"""
    
    def __init__(self, job_id: str, audio_file: Path, model: str = "base"):
        self.id = job_id
        self.audio_file = audio_file
        self.model = model
        self.status = "pending"
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.error = None
        self.remote_audio_path = None
        self.remote_output_path = None
        self.result = None
        self.pid = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "audio_file": str(self.audio_file),
            "model": self.model,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "remote_audio_path": self.remote_audio_path,
            "remote_output_path": self.remote_output_path,
            "pid": self.pid
        }


class JobManager:
    """Manages transcription jobs on GPU instances"""
    
    def __init__(self, ssh_connection: SSHConnection):
        self.ssh_connection = ssh_connection
        self.jobs = {}
        self.job_file = Path.home() / ".xtotext" / "transcription_jobs.json"
        self.job_file.parent.mkdir(exist_ok=True)
        self._load_jobs()
        
    def submit_job(self, audio_file: Path, model: str = "base", use_faster_whisper: bool = True) -> TranscriptionJob:
        """Submit a new transcription job"""
        job_id = str(uuid.uuid4())[:8]
        job = TranscriptionJob(job_id, audio_file, model)
        
        print(f"\n=== Submitting job {job_id} ===")
        print(f"Audio file: {audio_file.name}")
        print(f"Model: {model}")
        
        # Create remote directories
        self.ssh_connection.execute_command(
            f"mkdir -p {REMOTE_WORKSPACE} {REMOTE_AUDIO_DIR} {REMOTE_OUTPUT_DIR}"
        )
        
        # Upload audio file
        job.remote_audio_path = f"{REMOTE_AUDIO_DIR}/{job_id}_{audio_file.name}"
        job.remote_output_path = f"{REMOTE_OUTPUT_DIR}/{job_id}_transcript.json"
        
        print("Uploading audio file...")
        if not self.ssh_connection.upload_file(audio_file, job.remote_audio_path):
            job.status = "failed"
            job.error = "Failed to upload audio file"
            self.jobs[job_id] = job
            self._save_jobs()
            return job
            
        # Create transcription script
        if use_faster_whisper:
            script = self._create_faster_whisper_script(job)
        else:
            script = self._create_whisper_script(job)
            
        script_path = f"{REMOTE_WORKSPACE}/job_{job_id}.py"
        self.ssh_connection.create_remote_script(script, script_path)
        
        # Submit job in background using nohup
        cmd = f"cd {REMOTE_WORKSPACE} && nohup python3 job_{job_id}.py > job_{job_id}.log 2>&1 & echo $!"
        ret, out, err = self.ssh_connection.execute_command(cmd)
        
        if ret == 0 and out.strip().isdigit():
            job.pid = int(out.strip())
            job.status = "running"
            job.started_at = datetime.now()
            print(f"Job submitted with PID: {job.pid}")
        else:
            job.status = "failed"
            job.error = f"Failed to submit job: {err}"
            
        self.jobs[job_id] = job
        self._save_jobs()
        return job
        
    def check_job(self, job_id: str) -> TranscriptionJob:
        """Check the status of a job"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
            
        job = self.jobs[job_id]
        
        if job.status not in ["running", "pending"]:
            return job
            
        # Check if process is still running
        if job.pid:
            ret, out, err = self.ssh_connection.execute_command(
                f"ps -p {job.pid} > /dev/null 2>&1 && echo 'running' || echo 'done'"
            )
            
            if out.strip() == "done":
                # Process finished, check for output
                ret, out, err = self.ssh_connection.execute_command(
                    f"test -f {job.remote_output_path} && echo 'exists' || echo 'missing'"
                )
                
                if out.strip() == "exists":
                    job.status = "completed"
                    job.completed_at = datetime.now()
                else:
                    # Check log for errors
                    log_path = f"{REMOTE_WORKSPACE}/job_{job_id}.log"
                    ret, out, err = self.ssh_connection.execute_command(f"tail -50 {log_path}")
                    job.status = "failed"
                    job.error = f"No output file found. Log: {out}"
                    
                self._save_jobs()
                
        return job
        
    def get_job_log(self, job_id: str) -> str:
        """Get the log output for a job"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
            
        log_path = f"{REMOTE_WORKSPACE}/job_{job_id}.log"
        ret, out, err = self.ssh_connection.execute_command(f"cat {log_path} 2>/dev/null || echo 'Log not found'")
        return out
        
    def download_result(self, job_id: str, output_dir: Path) -> Optional[Path]:
        """Download the transcription result"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
            
        job = self.jobs[job_id]
        
        if job.status != "completed":
            print(f"Job {job_id} is not completed (status: {job.status})")
            return None
            
        output_dir.mkdir(exist_ok=True)
        local_path = output_dir / f"{job.audio_file.stem}_transcript.json"
        
        print(f"Downloading transcript for job {job_id}...")
        if self.ssh_connection.download_file(job.remote_output_path, local_path):
            # Add metadata
            with open(local_path, 'r') as f:
                transcript = json.load(f)
                
            transcript['metadata'] = {
                'job_id': job_id,
                'audio_file': job.audio_file.name,
                'model': job.model,
                'processing_time': (job.completed_at - job.started_at).total_seconds() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None
            }
            
            with open(local_path, 'w') as f:
                json.dump(transcript, f, indent=2)
                
            return local_path
        else:
            print(f"Failed to download result for job {job_id}")
            return None
            
    def cleanup_job(self, job_id: str):
        """Clean up remote files for a job"""
        if job_id not in self.jobs:
            return
            
        job = self.jobs[job_id]
        
        # Remove remote files
        self.ssh_connection.execute_command(
            f"rm -f {job.remote_audio_path} {job.remote_output_path} "
            f"{REMOTE_WORKSPACE}/job_{job_id}.py {REMOTE_WORKSPACE}/job_{job_id}.log"
        )
        
        print(f"Cleaned up files for job {job_id}")
        
    def list_jobs(self) -> List[TranscriptionJob]:
        """List all jobs"""
        # Update status of running jobs
        for job_id, job in self.jobs.items():
            if job.status == "running":
                self.check_job(job_id)
                
        return list(self.jobs.values())
        
    def monitor_jobs(self, interval: int = 10) -> Dict[str, TranscriptionJob]:
        """Monitor all running jobs"""
        running_jobs = {
            job_id: job for job_id, job in self.jobs.items() 
            if job.status == "running"
        }
        
        if not running_jobs:
            print("No running jobs to monitor")
            return {}
            
        print(f"\nMonitoring {len(running_jobs)} running jobs...")
        
        for job_id in running_jobs:
            self.check_job(job_id)
            job = self.jobs[job_id]
            print(f"  Job {job_id}: {job.status}")
            
        return running_jobs
        
    def _create_faster_whisper_script(self, job: TranscriptionJob) -> str:
        """Create faster-whisper transcription script"""
        return f'''#!/usr/bin/env python3
import json
import time
from faster_whisper import WhisperModel

print("Loading model {job.model}...")
start_time = time.time()

# Initialize model with GPU
model = WhisperModel("{job.model}", device="cuda", compute_type="float16")

print("Model loaded in {{:.1f}} seconds".format(time.time() - start_time))
print("Starting transcription of {job.audio_file.name}...")

# Run transcription
segments, info = model.transcribe(
    "{job.remote_audio_path}",
    beam_size=5,
    word_timestamps=True,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500)
)

print(f"Language: {{info.language}} (probability: {{info.language_probability:.2f}})")
print(f"Duration: {{info.duration:.2f}} seconds")

# Convert to list and format
result = {{
    "language": info.language,
    "duration": info.duration,
    "language_probability": info.language_probability,
    "segments": []
}}

segment_count = 0
for segment in segments:
    result["segments"].append({{
        "id": segment.id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "words": [
            {{
                "start": word.start,
                "end": word.end,
                "word": word.word,
                "probability": word.probability
            }}
            for word in (segment.words or [])
        ]
    }})
    segment_count += 1
    if segment_count % 100 == 0:
        print(f"Processed {{segment_count}} segments...")

print(f"Total segments: {{segment_count}}")
print(f"Transcription completed in {{time.time() - start_time:.1f}} seconds")

# Save results
with open("{job.remote_output_path}", "w") as f:
    json.dump(result, f, indent=2)

print("Transcript saved to {job.remote_output_path}")
'''
        
    def _create_whisper_script(self, job: TranscriptionJob) -> str:
        """Create standard whisper transcription script"""
        return f'''#!/usr/bin/env python3
import json
import time
import whisper

print("Loading model {job.model}...")
start_time = time.time()

# Load model
model = whisper.load_model("{job.model}")

print("Model loaded in {{:.1f}} seconds".format(time.time() - start_time))
print("Starting transcription of {job.audio_file.name}...")

# Transcribe
result = model.transcribe(
    "{job.remote_audio_path}",
    verbose=False,
    word_timestamps=True
)

print(f"Language: {{result['language']}}")
print(f"Transcription completed in {{time.time() - start_time:.1f}} seconds")

# Save results
with open("{job.remote_output_path}", "w") as f:
    json.dump(result, f, indent=2)

print("Transcript saved to {job.remote_output_path}")
'''
        
    def _save_jobs(self):
        """Save jobs to file"""
        data = {
            job_id: job.to_dict() 
            for job_id, job in self.jobs.items()
        }
        
        with open(self.job_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def _load_jobs(self):
        """Load jobs from file"""
        if not self.job_file.exists():
            return
            
        try:
            with open(self.job_file, 'r') as f:
                data = json.load(f)
                
            for job_id, job_data in data.items():
                job = TranscriptionJob(
                    job_id,
                    Path(job_data['audio_file']),
                    job_data['model']
                )
                job.status = job_data['status']
                job.created_at = datetime.fromisoformat(job_data['created_at'])
                if job_data.get('started_at'):
                    job.started_at = datetime.fromisoformat(job_data['started_at'])
                if job_data.get('completed_at'):
                    job.completed_at = datetime.fromisoformat(job_data['completed_at'])
                job.error = job_data.get('error')
                job.remote_audio_path = job_data.get('remote_audio_path')
                job.remote_output_path = job_data.get('remote_output_path')
                job.pid = job_data.get('pid')
                
                self.jobs[job_id] = job
        except Exception as e:
            print(f"Failed to load jobs: {e}")