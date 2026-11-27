import os
import stat
import tempfile
from pathlib import Path
from threading import Lock

from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor

from marker.ocr import IndraRecognitionPredictor


_surya_cache_lock = Lock()
_surya_cache_path: Path | None = None


def _check_writable_cache(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=path):
        pass
    return path


def _surya_cache_dir() -> Path:
    """Keep Surya's locks, sentinels, and logs in one directory per process."""
    global _surya_cache_path
    with _surya_cache_lock:
        if _surya_cache_path is None:
            default = Path.home() / ".cache" / "datalab" / "surya"
            try:
                _surya_cache_path = _check_writable_cache(default)
            except OSError:
                configured = os.environ.get("XDG_CACHE_HOME")
                if configured:
                    fallback = (
                        Path(configured).expanduser().absolute() / "datalab" / "surya"
                    )
                else:
                    get_uid = getattr(os, "getuid", None)
                    uid = get_uid() if get_uid else "user"
                    fallback = Path(tempfile.gettempdir()) / f"datalab-surya-{uid}"
                    fallback.mkdir(exist_ok=True, mode=0o700)
                    entry = fallback.stat(follow_symlinks=False)
                    if get_uid and (
                        not stat.S_ISDIR(entry.st_mode)
                        or entry.st_uid != uid
                        or stat.S_IMODE(entry.st_mode) & 0o077
                    ):
                        raise PermissionError(
                            f"Surya cache must be a private directory: {fallback}"
                        )
                _surya_cache_path = _check_writable_cache(fallback)
        return _surya_cache_path


def _configure_surya_cache() -> None:
    # Surya 0.24 imports the cache resolver by name in these local backends.
    from surya.inference.backends import llamacpp, mlx, spawn

    spawn._cache_dir = llamacpp._cache_dir = mlx._cache_dir = _surya_cache_dir


def create_inference_manager(
    inference_backend: str | None = None, *, lazy: bool = True
) -> SuryaInferenceManager:
    """Build a Surya manager with Marker's server cache resolver."""
    _configure_surya_cache()
    return SuryaInferenceManager(method=inference_backend, lazy=lazy)


def create_model_dict(
    device=None,
    dtype=None,
    attention_implementation: str | None = None,
    inference_manager: SuryaInferenceManager | None = None,
    inference_backend: str | None = None,
    load_indra: bool = True,
    no_server: bool = False,
) -> dict:
    """Build Marker predictors. With ``no_server``, only Indra is available.

    ``device="cpu"`` selects the CPU execution provider for Indra. Configure
    the Surya backend separately. Marker does not set its dtype or attention.
    """
    if dtype is not None or attention_implementation is not None:
        raise ValueError("Marker does not support dtype or attention_implementation.")
    if device is not None and (
        not load_indra or str(device) not in {"cpu", "cpu:0"}
    ):
        raise ValueError("Marker supports an explicit CPU device for Indra only.")
    if no_server and (inference_manager is not None or inference_backend is not None):
        raise ValueError("no_server cannot accept a Surya manager or backend.")
    if inference_manager is not None and inference_backend is not None:
        raise ValueError("Specify either inference_manager or inference_backend.")

    manager = None
    if not no_server:
        if inference_manager is None:
            manager = create_inference_manager(inference_backend)
        else:
            _configure_surya_cache()
            manager = inference_manager
    indra = (
        IndraRecognitionPredictor(
            providers=["CPUExecutionProvider"] if device is not None else None
        )
        if load_indra
        else None
    )
    rec = RecognitionPredictor(manager) if manager is not None else None
    return {
        "inference_manager": manager,
        "recognition_model": rec,
        "layout_model": indra,
        "fast_layout_model": indra,
        "ocr_error_model": None,
        **({"indra_model": indra} if load_indra else {}),
    }


def shutdown_models(model_dict: dict) -> None:
    """Stop the shared inference server if this process spawned it."""
    manager = model_dict.get("inference_manager")
    if manager is not None:
        manager.stop()
