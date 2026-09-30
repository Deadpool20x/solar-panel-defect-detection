from pathlib import Path

labels_dir = Path("/kaggle/working/b2_dataset_clean/test/labels")
counts = []
for lbl_path in labels_dir.glob("*.txt"):
    lines = lbl_path.read_text().strip().splitlines()
    dusty_boxes = sum(1 for l in lines if l.split()[0] == "1")
    if dusty_boxes > 0:
        counts.append((dusty_boxes, lbl_path.stem))

counts.sort(reverse=True)
for n, name in counts[:10]:
    print(n, name)