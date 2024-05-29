from tqdm import tqdm
import multiprocessing
import fadtk
import os
import torchaudio
import numpy as np
from fadtk.fad import FrechetAudioDistance
from fadtk import Path


def split_to_overlapping_windows(ptrns, inp_path, out_path, overlap=0.1, save_dir_structure=True):
    os.makedirs(out_path, exist_ok=True)
    
    for ptrn in ptrns:
        for f in Path(inp_path).glob(ptrn):
            aud, sr = torchaudio.load(f)
            for i, j in enumerate(range(0, aud.shape[1], int(10*( 1 - overlap)) * sr)):
            # now save it with keeping directory structure
                if save_dir_structure:
                    os.makedirs(os.path.join(out_path, str(f.parent).replace(inp_path, '')), exist_ok=True)
                    torchaudio.save(os.path.join(out_path, str(f.parent).replace(inp_path, ''), f'{f.stem}_{i}.wav'), aud[:, j:j+10*sr], sr)
                else:
                    torchaudio.save(os.path.join(out_path, f'{f.stem}_{i}.wav'), aud[:, j:j+10*sr], sr)

"""Code below adapted from: https://github.com/microsoft/fadtk"""

def _cache_embedding_batch(args):
    fs, ml, kwargs = args
    fad = FrechetAudioDistance(ml, **kwargs)
    for f in tqdm(fs):
        fad.cache_embedding_file(f)

def cache_embedding_files(files: Path, ml: fadtk.ModelLoader, workers: int = 8, pattern='*.*', **kwargs):
    """
    Get embeddings for all audio files in a directory.

    :param ml_fn: A function that returns a ModelLoader instance.
    """
    files = list(Path(files).glob(pattern))
    files = [f for f in files if 'orig' not in f.name and 'convert' not in str(f.parent)]

    # Filter out files that already have embeddings
    print(fadtk.get_cache_embedding_path(ml.name, files[0]))
    files = [f for f in files if not fadtk.get_cache_embedding_path(ml.name, f).exists()]
    if len(files) == 0:
        print("All files already have embeddings, skipping.")
        return
    
    print(f"[Frechet Audio Distance] Loading {len(files)} audio files...")
    # Split files into batches
    batches = list(np.array_split(files, workers))

    # Cache embeddings in parallel
    multiprocessing.set_start_method('spawn', force=True)
    for b in batches:
        # print(b)
        _cache_embedding_batch((b, ml, {}))
    # with torch.multiprocessing.Pool(workers) as pool:
        # pool.map(_cache_embedding_batch, [(b, ml, {}) for b in batches])
