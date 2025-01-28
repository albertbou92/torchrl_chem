import warnings

import numpy as np

import torch

from rdkit.Chem import AllChem as Chem, Draw


def get_mol(smiles_or_mol):
    """Converts a SMILES, RDKitMol, or None into RDKitMol or None."""
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    elif isinstance(smiles_or_mol, Chem.rdchem.Mol):
        return smiles_or_mol
    else:
        return None


def fraction_valid(mol_list):
    """Compute the fraction of valid molecules in a list of SMILES strings or Mols."""
    parsed_mols = []
    for mol in mol_list:
        mol = get_mol(mol)
        if mol:
            parsed_mols.append(1)
        else:
            parsed_mols.append(0)
    return np.mean(parsed_mols)


def randomize_smiles(smiles, random_type="restricted"):
    """Randomize a SMILES string using restricted or unrestricted randomization."""
    mol = get_mol(smiles)
    if mol:
        if random_type == "restricted":
            new_atom_order = list(range(mol.GetNumAtoms()))
            np.random.shuffle(new_atom_order)
            random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
            return Chem.MolToSmiles(random_mol, canonical=False)
        elif random_type == "unrestricted":
            return Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        else:
            raise ValueError(f"Invalid randomization type: {random_type}")
    else:
        warnings.warning(f"Could not randomize SMILES string: {smiles}")
        return smiles


def draw(mol_list, molsPerRow=5, subImgSize=(300, 300)):
    """Create a grid image of molecules from a list of SMILES strings or Mols."""
    mols = [get_mol(mol) for mol in mol_list]
    image = Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=subImgSize)

    return image


def get_fp(mol):
    """Create a Circular/Path based fingerprint from a SMILES string or RDKitMol."""
    mol = get_mol(mol)
    if mol:
        ecfp = Chem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=256)
        rdk = Chem.RDKFingerprint(mol, maxPath=6, fpSize=256, nBitsPerHash=2)
        fp = torch.cat([torch.tensor(ecfp), torch.tensor(rdk)])
        return fp
    else:
        return torch.zeros((512), dtype=torch.int64)


def get_fp_hist(mols):
    """Compute the histogram of fingerprints from a list of SMILES strings or Mols."""
    fp_hist = torch.vstack([get_fp(mol) for mol in mols]).sum(0)
    return fp_hist
