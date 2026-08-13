import os
import shutil
import time

src_dir = "data/vector_store"
timestamp = time.strftime("%Y%m%d_%H%M%S")
dst_dir = f"data/vector_store_backup_phase5a8_{timestamp}"

print(f"=== STEP 3: BACKUP VECTOR STORE ===")
print(f"Backing up '{src_dir}' -> '{dst_dir}'")

shutil.copytree(src_dir, dst_dir)

dst_files = os.listdir(dst_dir)
print("Backup directory created successfully. Files in backup:")
for f in dst_files:
    fp = os.path.join(dst_dir, f)
    print(f"  - {f} ({os.path.getsize(fp)} bytes)")

assert "index.faiss" in dst_files, "Backup missing index.faiss!"
assert "index.pkl" in dst_files, "Backup missing index.pkl!"
print("BACKUP VERIFICATION: PASSED CLEANLY")
