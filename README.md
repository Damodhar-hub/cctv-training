# CCTV Training Pipeline — YOLOv8s on IDD Detection

Fine-tunes YOLOv8s on the Indian Driving Dataset (IDD) Detection for Indian-specific
object detection (auto-rickshaws, two-wheelers, animals, etc.) used in C-Trace forensic
CCTV analysis tools.

## Classes (9)

| ID | Class          | IDD Sources                              |
|----|----------------|------------------------------------------|
| 0  | person         | person, rider                            |
| 1  | bicycle        | bicycle                                  |
| 2  | motorcycle     | motorcycle                               |
| 3  | car            | car                                      |
| 4  | autorickshaw   | autorickshaw, auto rickshaw, auto-rickshaw |
| 5  | bus            | bus                                      |
| 6  | truck          | truck                                    |
| 7  | vehicle_other  | caravan, trailer, vehicle fallback        |
| 8  | animal         | animal                                   |

## RunPod Setup

**Pod config:**
- GPU: RTX 4090 (24GB VRAM)
- Template: PyTorch 2.8
- Container disk: 80GB
- Volume disk: 0GB (not needed, ephemeral is fine)
- Enable Jupyter

## Deployment Steps

Once SSH or Jupyter terminal is open on the pod:

```bash
# 1. Set your IDD download URL (get from IIIT Hyderabad portal)
export IDD_TOKEN_URL='paste-your-iiit-download-url-here'

# 2. Clone repo
git clone https://github.com/USERNAME/cctv-training.git
cd cctv-training

# 3. Download + convert dataset (~30 min)
bash setup_pod.sh

# 4. Quick sanity check — 2 epochs (~15 min)
bash sanity_check.sh

# 5. Full training inside tmux (so you can detach)
tmux new -s training
bash train.sh            # 50 epochs (~7 hours on RTX 4090)
# Ctrl+B then D to detach — safe to close terminal
```

## Resuming / Monitoring

```bash
# Re-attach to training session
tmux attach -t training

# Check GPU utilization
nvidia-smi
```

## Output

Best model location after training:
```
cctv_v2/indian_run1/weights/best.pt   (~22MB)
```

**To download:** Use Jupyter file browser (navigate to the path above) or:
```bash
scp root@<pod-ip>:/workspace/cctv_v2/indian_run1/weights/best.pt ./best.pt
```

## Files

| File                    | Purpose                                    |
|-------------------------|--------------------------------------------|
| `requirements.txt`     | Python dependencies                        |
| `convert_idd_to_yolo.py` | Converts IDD VOC XML → YOLO format       |
| `setup_pod.sh`         | One-command pod bootstrap (download+convert)|
| `sanity_check.sh`      | 2-epoch test run                           |
| `train.sh`             | Full 50-epoch training (run in tmux)       |
| `.gitignore`           | Prevents committing datasets/weights/tokens|
