import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Configuration
# FFT parameter sets
fft_configs = [
    {"n_fft": 2048, "hop_length": 512, "win_length": 2048},
    {"n_fft": 2048, "hop_length": 256, "win_length": 2048},
    {"n_fft": 1024, "hop_length": 160, "win_length": 1024},
    {"n_fft": 1024, "hop_length": 256, "win_length": 1024}
]

# NMF rank sweep
rank_list = [10, 20, 40, 80, 120]

# Fixed number of iterations (sensible default)
n_iter = 150

# Paths (adjust to your project structure)
base_dir = Path("/Users/jinhee/Downloads/Code/mlsp-bss")
main_script = base_dir / "scripts" / "nmf_separation.py"
input_file = base_dir / "audio" / "originals/Nude_mix.wav"
vocal_ref = base_dir / "audio" / "stems/Nude_vocal.wav"
instr_ref = base_dir / "audio" / "originals/Nude_instrumental.wav"

# Output directory for logs and plots
results_dir = base_dir / "tuning_results"
results_dir.mkdir(exist_ok=True)

# ==========================================================
# Run experiments
# ==========================================================
results = []

for fft_cfg in fft_configs:
    avg_sdrs = []
    for rank in rank_list:
        print(f"\n=== Running: FFT={fft_cfg['n_fft']}, Rank={rank} ===")

        cmd = [
            "python", str(main_script),
            "-i", str(input_file),
            "-v", str(vocal_ref),
            "-r", str(instr_ref),
            "--n-base", str(rank),
            "--n-iter", str(n_iter), 
            "--n-fft", str(fft_cfg["n_fft"]),
            "--hop-length", str(fft_cfg["hop_length"]),
            "--win-length", str(fft_cfg["win_length"]),
        ]

        # Capture stdout
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        
        # Try to find average SDR from the printed output
        avg_sdr = None
        for line in output.splitlines():
            if "Average SDR" in line:
                avg_sdr = float(line.split(":")[1].strip().split()[0])
                break

        if avg_sdr is not None:
            avg_sdrs.append(avg_sdr)
        else:
            avg_sdrs.append(np.nan)

        # Save raw output log
        log_path = results_dir / f"log_fft{fft_cfg['n_fft']}_rank{rank}.txt"
        with open(log_path, "w") as f:
            f.write(output)

    results.append({
        "fft": fft_cfg,
        "ranks": rank_list,
        "avg_sdrs": avg_sdrs
    })

# Save results to JSON
json_path = results_dir / "results_summary.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

plt.figure(figsize=(8,6))

for fft_cfg in fft_configs:
    fft_label = f"window length={fft_cfg['n_fft']}, hop size={fft_cfg['hop_length']}"
    subset = [r for r in results if r["fft"] == fft_cfg]
    for r in subset:
        plt.plot(
            r["ranks"],
            r["avg_sdrs"],
            marker="o",
            label=fft_label
        )

plt.title(f"NMF Parameter Sweep: Rank vs Average SDR")
plt.xlabel("NMF Rank (n_base)")
plt.ylabel("Average SDR (dB)")
plt.legend()
plt.tight_layout()

plot_path = results_dir / f"nmf_param_tuning.png"
plt.savefig(plot_path, dpi=200)
plt.show()