import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import mir_eval
from tqdm import tqdm
from ssspy.bss.ilrma import GaussILRMA

def normalize(x):
    """Normalize audio to prevent clipping"""
    m = np.max(np.abs(x))
    if m > 0:
        return x / m
    else:
        return x

def evaluate_separation(references, estimates):
    """
    Evaluate separation using BSS_Eval metrics

    Args:
        references: array of shape (n_sources, n_samples)
        estimates: array of shape (n_sources, n_samples)

    Returns:
        sdr, sir, sar, sisdr: SDR, SIR, SAR, and SI-SDR values
    """
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(references, estimates)
    return sdr, sir, sar, perm

def main(args):

    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    
    # configs
    n_base = args.n_base
    n_iter = args.n_iter
    n_fft = args.n_fft
    hop_length = args.hop_length
    win_length = args.win_length

    # Fix input path
    input_file = Path(args.input_file)
    if not input_file.is_absolute():
        input_file = base_dir / input_file
    singer, songname, _ = input_file.stem.split('_')

    output_dir = base_dir / "audio" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load audio
    audio, sr = librosa.load(input_file, sr=None, mono=False)
    audio = normalize(audio)
    
    # audio: (n_channels, n_samples)
    # Compute STFT per channel
    # (n_channels, n_freqs, n_frames)
    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True, window='hann')
    print(f"input shape: {spec.shape}") 

    # 2. Run ILRMA
    print("\nPerforming ILRMA Separation")
    ilrma_model = GaussILRMA(n_basis=n_base)
    Y = ilrma_model(spec, n_iter=n_iter)
    
    print(f"output shape: {Y.shape}") 

    # 3. ISTFT for each separated source
    sources_time = []
    for s in range(Y.shape[0]):
        y_time = librosa.istft(Y[s], hop_length=hop_length, win_length=win_length, center=True, window='hann')
        sources_time.append(y_time)
    source1, source2 = sources_time

    try:
        print("\nPerforming Evaluation")

        vocal_ref, _ = librosa.load(args.vocal_ref, sr=sr, mono=True)
        instr_ref, _ = librosa.load(args.instr_ref, sr=sr, mono=True)

        # Truncate to shortest length
        min_len = min(len(source1), len(source2), len(vocal_ref), len(instr_ref))
        source1 = source1[:min_len]
        source2 = source2[:min_len]
        vocal_ref = vocal_ref[:min_len]
        instr_ref = instr_ref[:min_len]

        # Normalize
        source1 = normalize(source1)
        source2 = normalize(source2)
        vocal_ref = normalize(vocal_ref)
        instr_ref = normalize(instr_ref)

        references = np.array([vocal_ref, instr_ref])
        estimates = np.array([source1, source2])
        sdr, sir, sar, perm = evaluate_separation(references, estimates)

        sources = [source1, source2]
        vocal_est = sources[perm[0]]
        instr_est = sources[perm[1]]
        
        # Save to target dir
        config_suffix = f"_nb{n_base}_ni{n_iter}_fft{n_fft}_hop{hop_length}"
        output_file1 = output_dir / f"{singer}_{songname}_{args.output_prefix}{config_suffix}_vocals.wav"
        output_file2 = output_dir / f"{singer}_{songname}_{args.output_prefix}{config_suffix}_instrumental.wav"
        sf.write(output_file1, vocal_est.T, sr)
        sf.write(output_file2, instr_est.T, sr)

        print(f"\nBest match: source{perm[0]+1}=vocals, source{perm[1]+1}=instrumental")

        print("\nBSS Metrics:")
        print(f"    Vocals:")
        print(f"        SDR: {sdr[0]:.2f} dB")
        print(f"        SIR: {sir[0]:.2f} dB")
        print(f"        SAR: {sar[0]:.2f} dB")
        print(f"    Instrumental:")
        print(f"        SDR: {sdr[1]:.2f} dB")
        print(f"        SIR: {sir[1]:.2f} dB")
        print(f"        SAR: {sar[1]:.2f} dB")
        print(f"\nAverage SDR: {np.mean(sdr):.2f} dB")
        print(f"\nAverage SIR: {np.mean(sir):.2f} dB")
        print(f"\nAverage SAR: {np.mean(sar):.2f} dB")
        
    except Exception as e:
        print(f"\nError: evaluation failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='nmf_separation', description='NMF-based blind source separation')
    parser.add_argument('-i', '--input-file',
                        help='Input audio file path')
    parser.add_argument('-o', '--output-prefix', 
                        default='ilrma_source',
                        help='Output file prefix (default: ilrma_source)')
    parser.add_argument('-v', '--vocal_ref', 
                        help='Vocal reference for evaluation')
    parser.add_argument('-r', '--instr_ref', 
                        help='Instrumental reference for evaluation')
    parser.add_argument('--n-base', type=int, default=40,
                        help='Number of basis components (default: 40)')
    parser.add_argument('--n-iter', type=int, default=100,
                        help='Number of algorithm iterations (default: 100)')
    parser.add_argument('--n-fft', type=int, default=2048,
                        help='FFT window size (default: 2048)')
    parser.add_argument('--hop-length', type=int, default=512,
                        help='Hop length between frames (default: 512)')
    parser.add_argument('--win-length', type=int, default=2048,
                        help='Window length for STFT (default: 2048)')
    args = parser.parse_args()

    main(args)
