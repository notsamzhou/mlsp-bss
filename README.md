# mlsp-bss
Investigation of Blind Source Separation Techniques for MLSP

## Setup

### Requirements
- Python 3.11+
- Dependencies listed in `requirements.txt`

### Installation
```bash
pip install -r requirements.txt
```

### Data
Uses stems of 'Nude' by Radiohead released by the band for a remix competition.

The recommended file structure is as follows:
```
mlsp-bss/
├── audio/
│   ├── stems/           # Individual tracks (vocal, drum, bass, guitar, synth)
│   ├── originals/       # Mixed audio (original + mix, instrumental created by script)
│   └── outputs/         # BSS separation outputs
├── scripts/
│   ├── create_groundtruth_audio.py
│   └── ica_separation.py
│   └── nmf_separation.py
│   └── tune_nmf_params.py
│   └── auxiva_separation.py.py
│   └── ilrma_separation.py
├── requirements.txt
└── README.md
```
## Usage
Generate ground truth mixes
```bash
python scripts/create_groundtruth_audio.py
```

ICA separation
```bash
python scripts/ica_separation.py -i {input file} -o {output file prefix} -v {vocal reference} -r {instrumental reference}
```

NMF separation
```bash
python scripts/nmf_separation.py -i {input file} -o {output file prefix} -v {vocal reference} -r {instrumental reference} --n-base {Number of basis components} --n-iter {algorithm iterations} --n-fft {FFT window size} --hop-length{FFT hop length} --win-length {STFT window length}
```

IVA separation
```bash
python scripts/ica_separation.py -i {input file} -o {output file prefix} -v {vocal reference} -r {instrumental reference} --n-iter {algorithm iterations} --n-fft {FFT window size} --hop-length{FFT hop length} --win-length {STFT window length}
```

ILRMA separation
```bash
python scripts/ilrma_separation.py -i {input file} -o {output file prefix} -v {vocal reference} -r {instrumental reference} --n-base {Number of basis components} --n-iter {algorithm iterations} --n-fft {FFT window size} --hop-length{FFT hop length} --win-length {STFT window length}
```

## Methods
Available separation methods include ICA, NMF, AuxIVA, and ILRMA.
