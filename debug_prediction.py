#!/usr/bin/env python3
"""
Interactive debugging session for prediction extraction
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from infrastructure.digital_ocean.simple_runner import SimpleDigitalOceanRunner


def main():
    print("="*80)
    print("PREDICTION DEBUGGING SESSION")
    print("="*80)
    print("Episode 8 - SMLR → $9 prediction debug")
    print("Timestamp: 52:56 (3176s)")
    print()
    
    # Create runner
    runner = SimpleDigitalOceanRunner()
    
    try:
        # Start DO droplet
        print("Starting Digital Ocean droplet...")
        runner.start(droplet_size="s-2vcpu-4gb", wait_time=60)
        
        # Upload code and scripts
        print("\nUploading code to Digital Ocean...")
        runner.upload_code(Path("src"))
        runner.upload_code(Path("config"))
        runner.upload_code(Path("scripts"))
        
        # Upload the Episode 8 transcript
        transcript_path = Path("data/episodes/bitcoin_dive_bar_analysis/transcripts/Bitcoin Dive Bar EP 8 - Tim B  Be Scarce - Sideways Forever BTC SMLR MSTR_yo6hikbIp5c_transcript.json")
        
        if not transcript_path.exists():
            print(f"Error: Transcript not found at {transcript_path}")
            return
            
        print(f"\nUploading Episode 8 transcript...")
        runner.ssh.upload_file(transcript_path, "/workspace/episode8_transcript.json")
        
        # Create a debug script on the remote
        debug_script = '''#!/usr/bin/env python3
import json
import sys
sys.path.insert(0, '/workspace/xtotext')
sys.path.insert(0, '/workspace/xtotext/src')

# Load transcript
with open('/workspace/episode8_transcript.json', 'r') as f:
    transcript_data = json.load(f)

# Find segments around timestamp 3176 (52:56)
target_time = 3176
segments = transcript_data.get('segments', [])

print("\\n" + "="*80)
print("SEGMENTS AROUND 52:56 (3176s)")
print("="*80)

# Find segments within 30 seconds of target
relevant_segments = []
for seg in segments:
    start = seg.get('start', 0)
    end = seg.get('end', 0)
    if abs(start - target_time) <= 30 or abs(end - target_time) <= 30:
        relevant_segments.append(seg)

# Print segments
for seg in relevant_segments:
    start = seg.get('start', 0)
    end = seg.get('end', 0)
    text = seg.get('text', '')
    
    # Convert to timestamp
    start_min = int(start // 60)
    start_sec = int(start % 60)
    end_min = int(end // 60)
    end_sec = int(end % 60)
    
    print(f"\\n[{start_min}:{start_sec:02d} - {end_min}:{end_sec:02d}] ({start:.1f}s - {end:.1f}s)")
    print(f"Text: {text}")

# Also get broader context (2 minutes around)
print("\\n" + "="*80)
print("BROADER CONTEXT (51:56 - 53:56)")
print("="*80)

context_segments = []
for seg in segments:
    start = seg.get('start', 0)
    if start >= 3116 and start <= 3236:  # 60 seconds before and after
        context_segments.append(seg)

full_text = " ".join([seg.get('text', '') for seg in context_segments])
print(f"\\nFull context text ({len(context_segments)} segments):")
print(full_text)

# Save context for LLM testing
with open('/workspace/episode8_context.txt', 'w') as f:
    f.write(full_text)
    
print("\\n\\nContext saved to /workspace/episode8_context.txt")
'''
        
        # Upload debug script
        runner.ssh.execute_command("cat > /workspace/debug_extract.py << 'EOF'\n" + debug_script + "\nEOF")
        runner.ssh.execute_command("chmod +x /workspace/debug_extract.py")
        
        # Run the debug script
        print("\nExtracting context around timestamp...")
        result = runner.run_command("cd /workspace && python3 debug_extract.py")
        
        if result['success']:
            print(result['stdout'])
        
        # Create interactive testing script
        test_script = '''#!/usr/bin/env python3
import json
import sys
import os
sys.path.insert(0, '/workspace/xtotext')
sys.path.insert(0, '/workspace/xtotext/src')
sys.path.insert(0, '/workspace/xtotext/config')

from llm.llm_client import OpenAIClient

# Load context
with open('/workspace/episode8_context.txt', 'r') as f:
    context_text = f.read()

print("\\n" + "="*80)
print("INTERACTIVE LLM TESTING")
print("="*80)
print(f"Context loaded: {len(context_text)} characters")
print("\\nYou can now test different prompts with test_prompt.py")

# Initialize client
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("\\nERROR: No OpenAI API key found!")
    print("Set it with: export OPENAI_API_KEY='your-key'")
    sys.exit(1)

client = OpenAIClient(api_key)

# Test with current prompt
print("\\nTesting with CURRENT system prompt...")
predictions = client.extract_predictions(context_text, {
    'title': 'Bitcoin Dive Bar EP 8',
    'date': '2025-07-23'
})

print(f"\\nFound {len(predictions)} predictions:")
for i, pred in enumerate(predictions):
    print(f"\\n{i+1}. {pred.get('asset')} → ${pred.get('price')}")
    print(f"   Quote: {pred.get('quote', 'N/A')}")
    print(f"   Speaker: {pred.get('speaker', 'N/A')}")
    print(f"   Confidence: {pred.get('confidence', 'N/A')}")
'''

        runner.ssh.execute_command("cat > /workspace/test_prompt.py << 'EOF'\n" + test_script + "\nEOF")
        runner.ssh.execute_command("chmod +x /workspace/test_prompt.py")
        
        # Setup Python environment
        print("\nSetting up Python environment...")
        runner.run_command("chmod +x /workspace/xtotext/scripts/digital_ocean/setup_environment.sh")
        runner.run_command("bash /workspace/xtotext/scripts/digital_ocean/setup_environment.sh")
        
        print("\n" + "="*80)
        print("DEBUGGING SESSION READY!")
        print("="*80)
        print(f"\nSSH into the droplet:")
        print(f"  ssh root@{runner.droplet_ip}")
        print("\nUseful commands:")
        print("  cd /workspace")
        print("  python3 debug_extract.py     # Re-extract context")
        print("  python3 test_prompt.py       # Test current prompt")
        print("  cat episode8_context.txt     # View extracted context")
        print("  nano /workspace/xtotext/src/llm/llm_client.py  # Edit prompt")
        print("\nThe SMLR $9 prediction timestamp context is ready for analysis!")
        print("\nPress Ctrl+C when done to destroy the droplet.")
        
        # Keep alive
        import time
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\nCleaning up...")
    finally:
        runner.cleanup()
        print("Droplet destroyed!")


if __name__ == "__main__":
    main()