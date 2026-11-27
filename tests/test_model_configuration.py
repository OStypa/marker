import os

import pytest
from surya.inference.backends import llamacpp, mlx, spawn

from marker import models
from marker.ocr import IndraRecognitionPredictor, select_recognition_model


@pytest.mark.parametrize("cache", ["home", "configured", "temporary"])
def test_surya_uses_one_cache_for_locks_and_backend_logs(cache, monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    configured = tmp_path / "configured"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(models, "_surya_cache_path", None)
    monkeypatch.setattr(models.tempfile, "gettempdir", lambda: str(tmp_path))
    for module in (spawn, llamacpp, mlx):
        monkeypatch.setattr(module, "_cache_dir", module._cache_dir)

    if cache != "home":
        (home / ".cache").touch()
    if cache == "configured":
        monkeypatch.setenv("XDG_CACHE_HOME", str(configured))

    attach = spawn.attach_or_spawn
    bundle = models.create_model_dict(inference_backend="llamacpp", load_indra=False)
    models.create_model_dict(inference_backend="llamacpp", load_indra=False)
    assert bundle["inference_manager"].method == "llamacpp"
    assert bundle["recognition_model"].manager is bundle["inference_manager"]
    assert spawn.attach_or_spawn is attach
    assert llamacpp.attach_or_spawn is attach
    assert mlx.attach_or_spawn is attach

    lock = spawn._lock_path("llamacpp")
    if cache == "home":
        assert lock.parent == home / ".cache" / "datalab" / "surya"
    elif cache == "configured":
        assert lock.parent == configured / "datalab" / "surya"
    else:
        assert lock.parent == tmp_path / f"datalab-surya-{os.getuid()}"
        assert lock.parent.stat().st_mode & 0o777 == 0o700
    assert os.environ["HOME"] == str(home)
    assert spawn._sentinel_path("llamacpp").parent == lock.parent
    assert llamacpp._cache_dir() == mlx._cache_dir() == lock.parent


def test_surya_reports_unwritable_configured_cache(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".cache").touch()
    configured = tmp_path / "configured"
    configured.touch()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(configured))
    monkeypatch.setattr(models, "_surya_cache_path", None)
    with pytest.raises(OSError):
        models._surya_cache_dir()


def test_no_server_uses_only_indra_and_rejects_conflicting_options(monkeypatch):
    def unexpected_manager(*args, **kwargs):
        pytest.fail("The no-server path requested a Surya manager.")

    monkeypatch.setattr(models, "create_inference_manager", unexpected_manager)
    bundle = models.create_model_dict(device="cpu", no_server=True)
    assert bundle["inference_manager"] is None
    assert bundle["recognition_model"] is None
    assert isinstance(bundle["indra_model"], IndraRecognitionPredictor)
    assert bundle["indra_model"].providers == ["CPUExecutionProvider"]
    assert (
        select_recognition_model(
            bundle["recognition_model"], bundle["indra_model"], mode="fast"
        )
        is bundle["indra_model"]
    )
    models.shutdown_models(bundle)

    for options in (
        {"device": "cuda"},
        {"dtype": "float32"},
        {"attention_implementation": "eager"},
        {"inference_backend": "llamacpp"},
        {"inference_manager": object()},
    ):
        with pytest.raises(ValueError):
            models.create_model_dict(no_server=True, **options)
