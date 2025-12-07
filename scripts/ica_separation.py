import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import mir_eval


def projection_back(Y, X):
    """
    Projects mono ICA sources back to the stereo microphone channels
    using Least Squares (Minimal Distortion Principle).
    
    Args:
        Y: Estimated sources (n_sources, n_samples)
        X: Original mixture (n_channels, n_samples)
        
    Returns:
        Y_images: Stereo images (n_sources, n_channels, n_samples)
    """
    n_sources, n_samples = Y.shape
    n_channels = X.shape[0]
    
    Y_images = np.zeros((n_sources, n_channels, n_samples))
    
    for s in range(n_sources):
        y_vec = Y[s, :]
        denom = np.dot(y_vec, y_vec) + 1e-10
        scales = (X @ y_vec) / denom
            
        Y_images[s, :, :] = np.outer(scales, y_vec)
            
    return Y_images

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
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(references, estimates)
    return sdr, sir, sar, perm

def flatten_stereo(stereo_audio):
    """
    Flattens stereo (2, N) audio into (2*N,) for evaluation
    """

    return stereo_audio.reshape(-1)


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

    # Apply ICA, project to stereo, and normalize
    H = ica(audio)
    H = projection_back(H, audio)
    source1 = normalize(H[0])
    source2 = normalize(H[1])

    # Save to target dir
    output_file1 = output_dir / f"{args.output_prefix}_source1.wav"
    output_file2 = output_dir / f"{args.output_prefix}_source2.wav"
    sf.write(output_file1, source1.T, sr)
    sf.write(output_file2, source2.T, sr)


    try:
        print("\nPerforming Evaluation")

        vocal_ref, _ = librosa.load(args.vocal_ref, sr=sr, mono=False)
        instr_ref, _ = librosa.load(args.instr_ref, sr=sr, mono=False)

        # Truncate to shortest length
        min_len = min(source1.shape[1], source2.shape[1], vocal_ref.shape[1], instr_ref.shape[1])
        source1 = source1[:, :min_len]
        source2 = source2[:, :min_len]
        vocal_ref = vocal_ref[:, :min_len]
        instr_ref = instr_ref[:, :min_len]

        # Normalize ground truths
        vocal_ref = normalize(vocal_ref)
        instr_ref = normalize(instr_ref)

        vocal_ref_flat = flatten_stereo(vocal_ref)
        instr_ref_flat = flatten_stereo(instr_ref)
        source1_flat = flatten_stereo(source1)
        source2_flat = flatten_stereo(source2)

        references = np.array([vocal_ref_flat, instr_ref_flat])
        estimates = np.array([source1_flat, source2_flat])
        sdr, sir, sar, perm = evaluate_separation(references, estimates)

        sources = [source1, source2]
        vocal_est = sources[perm[0]]
        instr_est = sources[perm[1]]

        print(f"\nBest match: source{perm[0]+1}=vocals, source{perm[1]+1}=instrumental")

        output_file1 = output_dir / f"{args.output_prefix}_vocals.wav"
        output_file2 = output_dir / f"{args.output_prefix}_instrumental.wav"
        sf.write(output_file1, vocal_est.T, sr)
        sf.write(output_file2, instr_est.T, sr)
        
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
    parser = argparse.ArgumentParser(prog='ica_separation', description='ICA-based blind source separation')
    parser.add_argument('-i', '--input-file',
                        default='audio/originals/Nude_mix.wav', 
                        help='Input audio file path (default: audio/originals/Nude_mix.wav)')
    parser.add_argument('-o', '--output-prefix', 
                        default='Nude_ica',
                        help='Output file prefix (default: Nude_ica)')
    parser.add_argument('-v', '--vocal_ref', 
                        default='audio/stems/Nude_vocal.wav',
                        help='Vocal reference for evaluation (default: audio/stems/Nude_vocal.wav)')
    parser.add_argument('-r', '--instr_ref', 
                        default='audio/originals/Nude_instrumental.wav',
                        help='Instrumental reference for evaluation (default: audio/originals/Nude_instrumental.wav)')
    args = parser.parse_args()

    main(args)
