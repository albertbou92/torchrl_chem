# Upgrade pip and install dependencies
python -m pip install --upgrade pip && \
python -m pip install \
    flake8 \
    pytest \
    pytest-cov \
    hydra-core \
    tqdm \
    torch \
    torchvision \
    transformers \
    promptsmiles \
    torchrl \
    rdkit==2023.09.03 \
    MolScore \
    deepsmiles \
    selfies \
    smi2sdf \
    smi2svg \
    atomInSmiles \
    safe-mol \
    molbloom

# Verify installations
python -c "import transformers; print(transformers.__version__)"
python -c "import promptsmiles"
# python -c "import mamba_ssm; print('mamba-ssm:', mamba_ssm.__version__)"  # Assuming mamba-ssm imports as mamba

# Install local package
cd ../acegen-open
pip install -e .
