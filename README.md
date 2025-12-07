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
python scripts/ica_separation.py
```

NMF separation
```bash
python scripts/nmf_separation.py
```

## Methods
Baseline methods include Independent Component Analysis and Non-negative Matrix Factorization with more to come.
