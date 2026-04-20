import shutil
import sys
from pathlib import Path


def _get_gensim_cache_dir(api_module):
    base_dir = getattr(api_module, "BASE_DIR", None)
    if base_dir:
        return Path(base_dir)
    return Path.home() / "gensim-data"


def _safe_unlink(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def _clear_cached_model_files(api_module, model_name):
    cache_root = _get_gensim_cache_dir(api_module)
    alt_name = model_name.replace("-", "_")

    candidates = [
        cache_root / model_name,
        cache_root / alt_name,
        cache_root / f"{model_name}.gz",
        cache_root / f"{alt_name}.gz",
        cache_root / "information.json",
    ]

    cleared_any = False
    for candidate in candidates:
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
            cleared_any = True
        elif candidate.exists() and _safe_unlink(candidate):
            cleared_any = True

    # Remove potentially stale dynamic modules loaded by gensim downloader.
    sys.modules.pop(model_name, None)
    sys.modules.pop(alt_name, None)
    return cleared_any


def _load_via_return_path(api_module, model_name):
    """Try loading keyed vectors from a downloaded file path."""
    from gensim.models import KeyedVectors

    model_path = api_module.load(model_name, return_path=True)

    # Preferred route for most gensim-data model files.
    try:
        return KeyedVectors.load(model_path, mmap="r")
    except Exception:
        # Last resort: try classic word2vec format.
        binary = str(model_path).endswith(".bin")
        return KeyedVectors.load_word2vec_format(model_path, binary=binary)


def load_pretrained_vectors_with_recovery(api_module, model_name, fallback_builder, print_fn=print):
    print_fn(f"사전학습 벡터 로드 시도: {model_name}")

    try:
        vectors = api_module.load(model_name)
        print_fn("사전학습 벡터 로드 성공")
        return vectors
    except AttributeError as exc:
        if "load_data" in str(exc):
            cleared = _clear_cached_model_files(api_module, model_name)
            if cleared:
                print_fn("손상된 로컬 gensim 캐시를 정리하고 다시 시도합니다")

            try:
                vectors = api_module.load(model_name)
                print_fn("사전학습 벡터 로드 성공 (재시도)")
                return vectors
            except Exception as retry_exc:  # noqa: BLE001
                # Some environments fail dynamic module loading but can still load
                # the downloaded file path directly.
                if "load_data" in str(retry_exc):
                    try:
                        vectors = _load_via_return_path(api_module, model_name)
                        print_fn("사전학습 벡터 로드 성공 (경로 우회 로딩)")
                        return vectors
                    except Exception as path_exc:  # noqa: BLE001
                        print_fn("사전학습 벡터 로드 실패 -> fallback Word2Vec 사용")
                        print_fn(f"원인: {path_exc}")
                        return fallback_builder()

                print_fn("사전학습 벡터 로드 실패 -> fallback Word2Vec 사용")
                print_fn(f"원인: {retry_exc}")
                return fallback_builder()

        print_fn("사전학습 벡터 로드 실패 -> fallback Word2Vec 사용")
        print_fn(f"원인: {exc}")
        return fallback_builder()
    except Exception as exc:  # noqa: BLE001
        print_fn("사전학습 벡터 로드 실패 -> fallback Word2Vec 사용")
        print_fn(f"원인: {exc}")
        return fallback_builder()
