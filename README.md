# Privacy project

Existing privacy-enhancing strategies typically secure all transmitted information without considering which attributes should be protected or how much protection is needed, leading to:
- Degraded target task performance  
- Unnecessary privacy overhead  
- Inability to handle heterogeneous privacy

## Features
- Local ADV training with client-specific surrogate models  
- Max-min optimization to balance utility/privacy  
- Dynamic $\lambda$ adjustment  
- Both homogeneous and heterogeneous privacy

## Repository Structure

```bash
SafeCoder/
├── dataset/                # Preprocessed three datasets
├── models/                 # Model definitions
├── utils/                  # Helper functions
├── configs/                # Config files for different datasets
└── README.md
