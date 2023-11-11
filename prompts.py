#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A collection of various PDB-related prompts
"""

# A list of abbreviated names and synonyms
# for various biophysical methonds
# that are typically used for integrative modeling

METHODS_KEYWORDS = {
    'CX-MS': [
        'cross-link', 'crosslink',
        'XL-MS', 'CX-MS', 'CL-MS', 'XLMS', 'CXMS', 'CLMS',
        "chemical crosslinking mass spectrometry",
        'photo-crosslinking', 'crosslinking restraints',
        'crosslinking-derived restraints', 'chemical crosslinking',
        'in vivo crosslinking', 'crosslinking data',
    ],

    'HDX': [
        'Hydrogen–deuterium exchange mass spectrometry',
        'Hydrogen/deuterium exchange mass spectrometry'
        'HDX', 'HDXMS', 'HDX-MS',
    ],

    'EPR': [
        'electron paramagnetic resonance spectroscopy',
        'EPR', 'DEER',
        "Double electron electron resonance spectroscopy",
    ],

    'FRET': [
        'FRET',
        "forster resonance energy transfer",
        "fluorescence resonance energy transfer",
    ],

    'AFM': [
        'AFM',  "atomic force microscopy",
    ],

    'SAS': [
        'SAS', 'SAXS', 'SANS', "Small angle solution scattering",
        "solution scattering", "SEC-SAXS", "SEC-SAS", "SASBDB",
        "Small angle X-ray scattering", "Small angle neutron scattering",
    ],

    '3DGENOME': [
        'HiC', 'Hi-C', "chromosome conformation capture",
    ],

    'Y2H': [
        'Y2H',
        "yeast two-hybrid",
    ],

    'DNA_FOOTPRINTING': [
        "DNA Footprinting",
        "hydroxyl radical footprinting",
    ],

    'XRAY_TOMOGRAPHY': [
        "soft x-ray tomography",
    ],

    'FTIR': [
        "FTIR", "Infrared spectroscopy",
        "Fourier-transform infrared spectroscopy",
    ],

    'FLUORESCENCE': [
        "Fluorescence imaging",
        "fluorescence microscopy", "TIRF",
    ],

    'EVOLUTION': [
        'coevolution', "evolutionary covariance",
    ],

    'PREDICTED': [
        "predicted contacts",
    ],

    'INTEGRATIVE': [
        "integrative structure", "hybrid structure",
        "integrative modeling", "hybrid modeling",
    ],

    'SHAPE': [
        'Hydroxyl Acylation analyzed by Primer Extension',
    ]
}


def keywords_dict_to_string(keywords: dict) -> str:
    """
    Convert dictionary with method keywords and synonyms
    to a string

    Example:

        keywords = {
            'AFM': [
                'AFM',  "atomic force microscopy",
            ],

            'SAS': [
                'SAS', "solution scattering",
            ],
        }

    Result:

        'AFM (AFM, atomic force microscopy) or SAS (SAS, solution scattering)'
    """

    methods_string = ''
    for i, (k, v) in enumerate(keywords.items()):
        if i > 0:
            methods_string += ' or '
        methods_string += f'{k} ({", ".join(v)})'

    return methods_string

def get_qbi_hackathon_prompt(keywords: dict) -> str:
    """
    Returns a prompt that was initially developed
    during the QBI Hackathon.
    """

    if len(keywords) == 0:
        raise(ValueError("Keywords dict can't be empty"))

    methods_string = keywords_dict_to_string(keywords)

    prompt = (
        "You are reading a materials and methods section "
        "of a scientific paper. "
        f"Here is the list of methods {methods_string}.\n\n"
        "Did the authors use any of them? "
        "Answer Yes or No, followed by the name(s) of methods. "
        "Use only abbreviations."
    )

    return prompt

if __name__ == '__main__':
    # Just call an example function
    print(get_qbi_hackathon_prompt(METHODS_KEYWORDS))
