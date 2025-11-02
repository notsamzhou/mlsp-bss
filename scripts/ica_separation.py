import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import mir_eval


def ica(M: np.ndarray):
    """
    Performs ICA using FOBI

    Args:
        M: Input matrix of shape (n_sources, n_samples)

    Returns:
        H: Independent sources of shape (n_sources, n_samples)
    """
    # Center the data
    M_centered = (M - M.mean(axis=1, keepdims=True))

    # Whiten the centered data
    U, s, _ = np.linalg.svd(M_centered @ M_centered.T)
    C = np.diag(1.0 / np.sqrt(s)) @ U.T
    X = C @ M_centered

    r = np.sum(X ** 2, axis=0)
    D_X = (X * r) @ X.T
    U_fobi, _, _ = np.linalg.svd(D_X)
    B = U_fobi.T
    H = B @ X

    return H


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
        sdr, sir, sar, perm: SDR, SIR, and SAR values
    """
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(references, estimates)
    return sdr, sir, sar



def main(args):

    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    # Fix input path
    input_file = Path(args.input_file)
    if not input_file.is_absolute():
        input_file = base_dir / input_file

    output_dir = base_dir / "audio" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load audio
    audio, sr = librosa.load(input_file, sr=None, mono=False)

    # Check if stereo
    if audio.ndim == 1 or audio.shape[0] != 2:
        print(f"\nError: stereo audio is expected i.e. shape=(2, n_samples). Got {audio.shape}")
        print("Aborting")
        return

    print("\nPerforming ICA Separation")

    # Apply ICA and normalize
    H = ica(audio)
    source1 = normalize(H[0])
    source2 = normalize(H[1])

    # Save to target dir
    output_file1 = output_dir / f"{args.output_prefix}1.wav"
    output_file2 = output_dir / f"{args.output_prefix}2.wav"
    sf.write(output_file1, source1, sr)
    sf.write(output_file2, source2, sr)


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

        # Normalize ground truths
        vocal_ref = normalize(vocal_ref)
        instr_ref = normalize(instr_ref)

        # Try both orderings to find which aligns better with vocal/instr
        references = np.array([vocal_ref, instr_ref])
        estimates_order1 = np.array([source1, source2])
        estimates_order2 = np.array([source2, source1])
        sdr1, sir1, sar1 = evaluate_separation(references, estimates_order1)
        sdr2, sir2, sar2 = evaluate_separation(references, estimates_order2)



        # Choose the ordering with better average SDR
        if np.mean(sdr1) > np.mean(sdr2):
            sdr, sir, sar = sdr1, sir1, sar1
            print("\nBest match: source1=vocals, source2=instrumental")
        else:
            sdr, sir, sar = sdr2, sir2, sar2
            print("\nBest match: source2=vocals, source1=instrumental")

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
        
    except Exception as e:
        print(f"\nError: evaluation failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ica_separation', description='ICA-based blind source separation')
    parser.add_argument('-i', '--input-file',
                        default='audio/originals/Nude_mix.wav', 
                        help='Input audio file path (default: audio/originals/Nude_mix.wav)')
    parser.add_argument('-o', '--output-prefix', 
                        default='Nude_ica_source',
                        help='Output file prefix (default: Nude_ica_source)')
    parser.add_argument('-v', '--vocal_ref', 
                        default='audio/stems/Nude_vocal.wav',
                        help='Vocal reference for evaluation (default: audio/stems/Nude_vocal.wav)')
    parser.add_argument('-r', '--instr_ref', 
                        default='audio/originals/Nude_instrumental.wav',
                        help='Instrumental reference for evaluation (default: audio/originals/Nude_instrumental.wav)')
    args = parser.parse_args()

    main(args)
