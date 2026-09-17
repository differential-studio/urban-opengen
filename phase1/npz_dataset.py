"""
training/npz_dataset.py

StyleGAN3-repo dataset class for the Urban OpenGen 4-channel tiles.npz
produced by phase0/reencode_tiles.py.

    x    : uint8 [N, 4, S, S]   footprint, height, street, green
    city : int16 [N]            city label, used for --cond=1

Arrays are loaded lazily in each worker process (the dataset object is pickled
to DataLoader workers, so we keep only the path in the pickle).

dihedral=True virtually multiplies the dataset by 8 (4 rotations x 2 flips).
All eight are legitimate city tiles, unlike faces, so this is real data
augmentation rather than the discriminator-only ADA kind. It subsumes xflip.
"""
import os
import numpy as np
from training.dataset import Dataset


class NpzDataset(Dataset):
    def __init__(self, path, resolution=None, dihedral=False, **super_kwargs):
        self._path = path
        if not os.path.isfile(path):
            raise IOError(f"{path} is not a file")
        with np.load(path) as d:
            shape = d["x"].shape           # np.load of an npz is lazy per key, this reads x once
            if "city" not in d:
                raise IOError("tiles.npz has no 'city' array")
        self._x = None
        self._city = None
        name = os.path.splitext(os.path.basename(os.path.dirname(os.path.abspath(path))))[0] or "tiles"
        raw_shape = list(shape)             # [N, C, H, W]
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError(f"tiles are {raw_shape[2]}x{raw_shape[3]}, not the requested {resolution}")
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
        self._dihedral = bool(dihedral)
        if self._dihedral:
            n = self._raw_idx.size
            self._raw_idx = np.tile(self._raw_idx, 8)
            self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)   # handled by _tform instead
            self._tform = np.repeat(np.arange(8, dtype=np.uint8), n)

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._dihedral:
            t = int(self._tform[idx])
            image = np.rot90(image, k=t % 4, axes=(1, 2))
            if t >= 4:
                image = image[:, :, ::-1]
            image = np.ascontiguousarray(image)
        elif self._xflip[idx]:
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def _ensure_loaded(self):
        if self._x is None:
            with np.load(self._path) as d:
                self._x = d["x"]
                self._city = d["city"].astype(np.int64)

    def _load_raw_image(self, raw_idx):
        self._ensure_loaded()
        return self._x[raw_idx]             # uint8 [C, H, W]

    def _load_raw_labels(self):
        self._ensure_loaded()
        return self._city                   # int64 [N], the base class one-hot encodes it

    def __getstate__(self):
        return dict(super().__getstate__(), _x=None, _city=None)
