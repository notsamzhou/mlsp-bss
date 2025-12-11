import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import mir_eval
from tqdm import tqdm

def nmf_separation(M, B_vocal_init=None, B_inst_init=None, W_vocal_init=None, W_inst_init=None, n_base=100, n_iter=500, eps=1e-10, random_seed=2025):
    """
    Perform blind source separation using Non-negative Matrix Factorization (NMF).

    Args:
        M: Input nonnegative magnitude spectrogram (array of shape (freq_bins, time_frames))
        B_vocal_init: Optional initial basis matrix for vocal source (default: random)
        B_inst_init: Optional initial basis matrix for instrumental source (default: random)
        W_vocal_init: Optional initial activation matrix for vocal source (default: random)
        W_inst_init: Optional initial activation matrix for instrumental source (default: random)
        n_base: Number of basis components per source (default: 100)
        n_iter: Number of NMF update iterations (default: 500)
        eps: Small constant to avoid division by zero (default: 1e-10)

    Returns:
        M_speech_rec: Reconstructed vocal spectrogram (array of shape (freq_bins, time_frames))
        M_music_rec: Reconstructed instrumental spectrogram (array of shape (freq_bins, time_frames))
    """
    assert np.all(M >= 0), "Matrix M has negative entries"
    
    p, q = M.shape
    np.random.seed(random_seed)
    if B_vocal_init is None:
        B_vocal_init = np.random.rand(p, n_base)
    if B_inst_init is None:
        B_inst_init = np.random.rand(p, n_base)
    if W_vocal_init is None:
        W_vocal_init = np.random.rand(n_base, q)
    if W_inst_init is None:
        W_inst_init = np.random.rand(n_base, q)
        
    B = np.concatenate([B_vocal_init, B_inst_init], axis=1)
    W = np.concatenate([W_vocal_init, W_inst_init], axis=0)
    
    assert np.all(B >= 0), "Matrix B has negative entries"
    assert np.all(W >= 0), "Matrix W has negative entries"
    
    for iter_i in tqdm(range(n_iter)):
        WH = B @ W + eps
        B *= (((M / WH) @ W.T) / ((np.ones_like(M) @ W.T) + eps))
        W *= ((B.T @ (M / WH)) / ((B.T @ np.ones_like(M)) + eps))
        
    split_point = W.shape[0] // 2
    W_vocal, W_inst = np.split(W, [split_point], axis=0)
    split_point = B.shape[1] // 2
    B_vocal, B_inst = np.split(B, [split_point], axis=1)
    
    M_speech_rec = B_vocal @ W_vocal
    M_music_rec = B_inst @ W_inst
    
    return M_speech_rec, M_music_rec

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
    def si_sdr(reference, estimate):
        ref_energy = np.sum(reference ** 2)
        if ref_energy == 0:
            return -np.inf
        scale = np.dot(reference, estimate) / ref_energy
        est_target = scale * reference
        e_noise = estimate - est_target
        return 10 * np.log10(np.sum(est_target ** 2) / np.sum(e_noise ** 2))
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(references, estimates)
    sisdr = []
    for r, e in zip(references, estimates):
        sisdr.append(si_sdr(r, e))
    return sdr, sir, sar, np.array(sisdr)

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

    output_dir = base_dir / "audio" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load audio
    audio, sr = librosa.load(input_file, sr=None, mono=True)  # mono 

    # Spectrogram
    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, center=False, win_length=win_length)
    mag = abs(spec)
    phase = spec / (mag + 2.2204e-16)

    print("\nPerforming NMF Separation")

    # Apply NMF and normalize
    mag_vocal_rec, mag_inst_rec = nmf_separation(mag, n_base=n_base, n_iter=n_iter)
    
    # soft mask
    eps_mask = 1e-10
    mag_total = mag_vocal_rec + mag_inst_rec
    mag_total_sq = mag_total ** 2
    mask_v = (mag_vocal_rec ** 2) / (mag_total_sq + eps_mask)
    mask_i = (mag_inst_rec ** 2) / (mag_total_sq + eps_mask)
    audio_vocal_rec = librosa.istft(mask_v * spec, hop_length=hop_length, center=False, win_length=win_length)
    audio_inst_rec = librosa.istft(mask_i * spec, hop_length=hop_length, center=False, win_length=win_length)
    
    source1 = normalize(audio_vocal_rec)
    source2 = normalize(audio_inst_rec)

    # Save to target dir
    config_suffix = f"_nb{args.n_base}_ni{args.n_iter}_fft{args.n_fft}_hop{args.hop_length}"
    output_file1 = output_dir / f"{args.output_prefix}{config_suffix}_1.wav"
    output_file2 = output_dir / f"{args.output_prefix}{config_suffix}_2.wav"
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
        sdr1, sir1, sar1, sisdr1 = evaluate_separation(references, estimates_order1)
        sdr2, sir2, sar2, sisdr2 = evaluate_separation(references, estimates_order2)

        # Choose the ordering with better average SDR
        if np.mean(sisdr1) > np.mean(sisdr2):
            sdr, sir, sar, sisdr = sdr1, sir1, sar1, sisdr1
            print("\nBest match: source1=vocals, source2=instrumental")
        else:
            sdr, sir, sar, sisdr = sdr2, sir2, sar2, sisdr2
            print("\nBest match: source2=vocals, source1=instrumental")

        print("\nBSS Metrics:")
        print(f"    Vocals:")
        print(f"        SI-SDR: {sisdr[0]:.2f} dB")
        print(f"        SDR: {sdr[0]:.2f} dB")
        print(f"        SIR: {sir[0]:.2f} dB")
        print(f"        SAR: {sar[0]:.2f} dB")
        print(f"    Instrumental:")
        print(f"        SI-SDR: {sisdr[1]:.2f} dB")
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
                        default='audio/originals/Nude_mix.wav', 
                        help='Input audio file path (default: audio/originals/Nude_mix.wav)')
    parser.add_argument('-o', '--output-prefix', 
                        default='Nude_nmf_source',
                        help='Output file prefix (default: Nude_ica_source)')
    parser.add_argument('-v', '--vocal_ref', 
                        default='audio/stems/Nude_vocal.wav',
                        help='Vocal reference for evaluation (default: audio/stems/Nude_vocal.wav)')
    parser.add_argument('-r', '--instr_ref', 
                        default='audio/originals/Nude_instrumental.wav',
                        help='Instrumental reference for evaluation (default: audio/originals/Nude_instrumental.wav)')
    parser.add_argument('--n-base', type=int, default=40,
                        help='Number of basis components (default: 40)')
    parser.add_argument('--n-iter', type=int, default=100,
                        help='Number of NMF iterations (default: 100)')
    parser.add_argument('--n-fft', type=int, default=2048,
                        help='FFT window size (default: 2048)')
    parser.add_argument('--hop-length', type=int, default=512,
                        help='Hop length between frames (default: 512)')
    parser.add_argument('--win-length', type=int, default=2048,
                        help='Window length for STFT (default: 2048)')
    args = parser.parse_args()

    main(args)
