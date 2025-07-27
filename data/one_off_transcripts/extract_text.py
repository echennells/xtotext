import json

with open("AMA： This Week's Guest Is YOU! ｜ Ep 29 ｜ S5_UTsBQ-aVx1s_transcript.json", "r") as f:
    data = json.load(f)
    
with open("AMA_clean_transcript.txt", "w") as f:
    f.write(data["text"])