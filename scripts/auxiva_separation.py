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
    Projects separated sources back to ALL microphones (Stereo preservation).
    
    Args:
        Y: Separated source STFTs (n_sources, n_freq, n_frames)
        X: Mixture STFTs (n_channels, n_freq, n_frames)
    
    Returns:
        Y_images: Stereo images of sources (n_sources, n_channels, n_freq, n_frames)
    """
    n_sources, n_freq, n_frames = Y.shape
    n_channels = X.shape[0]
    
    # Shape: (Source, Channel, Freq, Time)
    Y_images = np.zeros((n_sources, n_channels, n_freq, n_frames), dtype=np.complex64)
    
    for s in range(n_sources):
        for f in range(n_freq):
            y_vec = Y[s, f, :]
            
            # Project source s onto microphone c
            # scale = (y . x*) / (y . y*)
            denom = np.vdot(y_vec, y_vec).real + 1e-10
            numerators = X[:, f, :] @ y_vec.conj()
            scales = numerators / denom
            
            Y_images[s, :, f, :] = scales[:, None] * y_vec[None, :]
                
    return Y_images

def auxiva(X, n_iters=30):
    """
    Performs Blind Source Separation using AuxIVA (Iterative Projection).
    X shape: (n_channels, n_freq, n_frames)
    """
    n_channels, n_freq, n_frames = X.shape
    n_sources = n_channels 

    # 1. Initialize W
    # Shape: (Freq, Source, Channel)
    W = np.array([np.eye(n_sources, n_channels, dtype=np.complex64) for _ in range(n_freq)])
    
    # Initial separation: (Freq, Source, Time)
    Y = np.einsum('fsc,cft->fst', W, X)
    
    # Identity matrix for updates
    I = np.eye(n_sources)
    
    print(f"Running AuxIVA\n")
    
    for epoch in range(n_iters):
        # 2. Source Model (Variance Profile)
        # Sum over Frequencies (axis=0)
        # Result shape: (Source, Time)
        R = np.sqrt(np.sum(np.abs(Y)**2, axis=0)) 
        R[R < 1e-10] = 1e-10
        
        # 3. Update sources one by one
        for s in range(n_sources):
            # Contrast function derivative (weight)
            # Shape: (n_sources, n_frames) -> select source s -> (n_frames,)
            G = 1.0 / R[s, :]
            
            # Weighted Covariance Matrix V
            # Weight X by the inverse variance of the current source estimate
            # X shape: (Chan, Freq, Time). G shape: (Time,)
            # We need to broadcast G across frequencies.
            X_weighted = X * G[None, None, :] 
            
            # V shape: (Freq, Chan, Chan)
            V = np.einsum('cft, dft -> fcd', X_weighted, X.conj()) / n_frames
            
            # 4. Update Rule: (W @ V) @ w_new = e_s
            WV = np.matmul(W, V)
            
            # Create RHS vector: e_s
            rhs = np.tile(I[:, s][None, :, None], (n_freq, 1, 1))
            
            # Solve for w_new (column vector)
            w_new = np.linalg.solve(WV, rhs)
            
            # Squeeze to (Freq, Chan)
            w_new = w_new[..., 0]
            
            # 5. Normalize
            # Calculate w_new^H @ V @ w_new
            Vw = np.einsum('fcd, fd -> fc', V, w_new.conj())
            val = np.einsum('fc, fc -> f', w_new, Vw).real
            scale = np.sqrt(val) + 1e-10
            w_new = w_new / scale[:, None]
            
            # Update W
            W[:, s, :] = w_new.conj()
            
        # Recompute separation
        Y = np.einsum('fsc,cft->fst', W, X)

        if (epoch+1) % 10 == 0:
            print(f"  Iteration {epoch+1}/{n_iters} complete")
    
    # Transpose Y to (Sources, Freq, Time)
    Y = np.transpose(Y, (1, 0, 2))
            
    return Y

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

def flatten_stereo(stereo_audio):
    """
    Flattens stereo (2, N) audio into (2*N,) for evaluation
    """

    return stereo_audio.reshape(-1)

def main(args):

    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

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
    audio, sr = librosa.load(input_file, sr=None, mono=False)

    # Check if stereo
    if audio.ndim == 1 or audio.shape[0] != 2:
        print(f"\nError: stereo audio is expected i.e. shape=(2, n_samples). Got {audio.shape}")
        print("Aborting")
        return
    
    X = np.array([librosa.stft(audio[0], n_fft=n_fft, hop_length=hop_length),
                  librosa.stft(audio[1], n_fft=n_fft, hop_length=hop_length)])

    print("\nPerforming AuxIVA Separation")

    # Apply ICA and normalize
    Y_stft = auxiva(X, n_iters=n_iter)
    Y_stft = projection_back(Y_stft, X)

    src1_L = librosa.istft(Y_stft[0, 0], hop_length=hop_length, length=audio.shape[1])
    src1_R = librosa.istft(Y_stft[0, 1], hop_length=hop_length, length=audio.shape[1])
    source1 = np.vstack([src1_L, src1_R])

    src2_L = librosa.istft(Y_stft[1, 0], hop_length=hop_length, length=audio.shape[1])
    src2_R = librosa.istft(Y_stft[1, 1], hop_length=hop_length, length=audio.shape[1])
    source2 = np.vstack([src2_L, src2_R])

    source1 = normalize(source1)
    source2 = normalize(source2)

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

        # Try both orderings to find which aligns better with vocal/instr
        references = np.array([vocal_ref_flat, instr_ref_flat])
        estimates_order1 = np.array([source1_flat, source2_flat])
        estimates_order2 = np.array([source2_flat, source1_flat])
        sdr1, sir1, sar1 = evaluate_separation(references, estimates_order1)
        sdr2, sir2, sar2 = evaluate_separation(references, estimates_order2)



        # Choose the ordering with better average SDR
        if np.mean(sdr1) > np.mean(sdr2):
            vocal_est, instr_est = source1, source2
            sdr, sir, sar = sdr1, sir1, sar1
            print("\nBest match: source1=vocals, source2=instrumental")
        else:
            vocal_est, instr_est = source2, source1
            sdr, sir, sar = sdr2, sir2, sar2
            print("\nBest match: source2=vocals, source1=instrumental")

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
    parser = argparse.ArgumentParser(prog='auxiva_separation', description='AuxIVA-based blind source separation')
    parser.add_argument('-i', '--input-file',
                        default='audio/originals/Nude_mix.wav', 
                        help='Input audio file path (default: audio/originals/Nude_mix.wav)')
    parser.add_argument('-o', '--output-prefix', 
                        default='Nude_auxiva',
                        help='Output file prefix (default: Nude_ica)')
    parser.add_argument('-v', '--vocal_ref', 
                        default='audio/stems/Nude_vocal.wav',
                        help='Vocal reference for evaluation (default: audio/stems/Nude_vocal.wav)')
    parser.add_argument('-r', '--instr_ref', 
                        default='audio/originals/Nude_instrumental.wav',
                        help='Instrumental reference for evaluation (default: audio/originals/Nude_instrumental.wav)')
    
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
