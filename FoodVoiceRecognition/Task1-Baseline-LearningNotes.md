### 1. Notebook: 
CNN Baseline运行结果: [T1-CNN-Baseline.ipynb](https://github.com/frankyangdev/NLP-Learning/blob/main/FoodVoiceRecognition/T1-CNN-Baseline.ipynb)

### 2. Code Study:

#### unzip -qq

-q perform operations quietly (-qq = even quieter). Ordinarily unzip prints the names of the files it's extracting or testing, the extrac‐ tion methods, any file or zipfile comments that may be stored in the archive, and possibly a summary when finished with each archive. The -qq options suppress the printing of some or all of these messages.

#### Python Library: [librosa](https://librosa.org/doc/latest/index.html) 

librosa is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.

**Overview**


The *librosa* package is structured as collection of submodules:

  - librosa

    - :ref:`librosa.beat <beat>`
        Functions for estimating tempo and detecting beat events.

    - :ref:`librosa.core <core>`
        Core functionality includes functions to load audio from disk, compute various
        spectrogram representations, and a variety of commonly used tools for
        music analysis.  For convenience, all functionality in this submodule is
        directly accessible from the top-level `librosa.*` namespace.
        
    - :ref:`librosa.decompose <decompose>`
        Functions for harmonic-percussive source separation (HPSS) and generic
        spectrogram decomposition using matrix decomposition methods implemented in
        *scikit-learn*.

    - :ref:`librosa.display <display>`
        Visualization and display routines using `matplotlib`.  

    - :ref:`librosa.effects <effects>`
        Time-domain audio processing, such as pitch shifting and time stretching.
        This submodule also provides time-domain wrappers for the `decompose`
        submodule.

    - :ref:`librosa.feature <feature>`
        Feature extraction and manipulation.  This includes low-level feature
        extraction, such as chromagrams, Mel spectrogram, MFCC, and various other
        spectral and rhythmic features.  Also provided are feature manipulation
        methods, such as delta features and memory embedding.

    - :ref:`librosa.filters <filters>`
        Filter-bank generation (chroma, pseudo-CQT, CQT, etc.).  These are primarily
        internal functions used by other parts of *librosa*.

    - :ref:`librosa.onset <onset>`
        Onset detection and onset strength computation.

    - :ref:`librosa.segment <segment>`
        Functions useful for structural segmentation, such as recurrence matrix
        construction, time-lag representation, and sequentially constrained
        clustering.

    - :ref:`librosa.sequence <sequence>`
        Functions for sequential modeling.  Various forms of Viterbi decoding,
        and helper functions for constructing transition matrices.

    - :ref:`librosa.util <util>`
        Helper utilities (normalization, padding, centering, etc.)


