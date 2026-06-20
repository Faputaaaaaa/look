import os
import time

base = r'd:\deeplearning\learn\项目\experiments'

for i in range(1, 61):
    dirs = [d for d in os.listdir(base) if d.startswith('exp_20260609')]
    if not dirs:
        print(f"{time.strftime('%H:%M:%S')} - Check #{i} - No exp_20260609 dirs found")
        time.sleep(60)
        continue
    
    latest = max(dirs)
    log_path = os.path.join(base, latest, 'training_log.csv')
    
    if os.path.exists(log_path):
        full_path = os.path.join(base, latest)
        print(f"TRAINING_COMPLETE:{full_path}")
        break
    
    print(f"{time.strftime('%H:%M:%S')} - Check #{i} - {latest} - still training...")
    time.sleep(60)
else:
    print("TIMEOUT: Training did not complete within 60 minutes of monitoring")
