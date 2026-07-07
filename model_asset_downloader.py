import concurrent.futures
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from typing import Optional
from urllib.parse import parse_qs, urlparse

import requests

import folder_paths

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

CIVITAI_API_BASE = "https://civitai.com/api"
HF_BASE = "https://huggingface.co"

CIVITAI_TOKEN_ENV = "CIVITAI_API_TOKEN"
HF_TOKEN_ENV = "HF_TOKEN"

SAVE_TYPES = [
    "checkpoints",
    "loras",
    "vae",
    "text_encoders",
    "controlnet",
    "embeddings",
    "upscale_models",
    "unet",
    "clip",
    "clip_vision",
]

# folder_paths key(s) to try, in order, for a given save_type. Older ComfyUI
# installs only register "clip" for text-encoder models; newer ones also
# register "text_encoders" pointing at the same directory.
SAVE_TYPE_FOLDER_ALIASES = {
    "text_encoders": ["text_encoders", "clip"],
}

CHUNK_SIZE = 1024 * 1024  # 1 MiB
DEFAULT_MAX_CONNECTIONS = 4
MAX_ALLOWED_CONNECTIONS = 8
MIN_SIZE_FOR_PARALLEL = 32 * 1024 * 1024  # below this, one connection is enough


class AssetDownloadError(Exception):
    pass


def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name.strip().replace("\\", "/"))
    name = re.sub(r"[^A-Za-z0-9._\-\(\)\[\] ]", "_", name)
    name = name.strip(". ")
    if not name:
        raise AssetDownloadError("Could not determine a valid filename.")
    return name


def _get_save_dir(save_type: str) -> str:
    candidates = SAVE_TYPE_FOLDER_ALIASES.get(save_type, [save_type])
    for name in candidates:
        try:
            paths = folder_paths.get_folder_paths(name)
        except KeyError:
            continue
        if paths:
            save_dir = paths[0]
            os.makedirs(save_dir, exist_ok=True)
            return save_dir
    raise AssetDownloadError(f"No configured directory for save_type: {save_type}")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_cache_path(dest_path: str) -> str:
    return dest_path + ".dlcache.json"


def _load_hash_cache(dest_path: str):
    try:
        with open(_hash_cache_path(dest_path), "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _save_hash_cache(dest_path: str, size: int, mtime: float, sha256: str):
    try:
        with open(_hash_cache_path(dest_path), "w") as f:
            json.dump({"size": size, "mtime": mtime, "sha256": sha256}, f)
    except OSError:
        pass


class _CivitaiSource:
    """Resolves a Civitai URL into concrete download metadata."""

    MODEL_PAGE_RE = re.compile(r"^/models/(\d+)")
    DOWNLOAD_RE = re.compile(r"^/api/download/models/(\d+)")

    def __init__(self, token: Optional[str]):
        self.token = token

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _get_json(self, url: str):
        try:
            resp = requests.get(url, headers=self._headers(), timeout=30)
        except requests.RequestException as e:
            raise AssetDownloadError(f"Failed to reach Civitai API ({url}): {e}")
        if resp.status_code == 401:
            raise AssetDownloadError(
                f"Civitai returned 401 Unauthorized for {url}. "
                f"Set the {CIVITAI_TOKEN_ENV} environment variable."
            )
        resp.raise_for_status()
        return resp.json()

    def resolve(self, url: str):
        parsed = urlparse(url)
        version_id = None

        m = self.DOWNLOAD_RE.match(parsed.path)
        if m:
            version_id = m.group(1)

        m = self.MODEL_PAGE_RE.match(parsed.path)
        if m:
            qs = parse_qs(parsed.query)
            if "modelVersionId" in qs:
                version_id = qs["modelVersionId"][0]
            else:
                version_id = self._latest_version_id(m.group(1))

        if version_id is None:
            raise AssetDownloadError(
                "Could not determine a Civitai model version from the URL. "
                "Use a model page URL (ideally with ?modelVersionId=...) or a "
                "direct '/api/download/models/<id>' URL."
            )

        return self._version_info(version_id)

    def _latest_version_id(self, model_id: str) -> str:
        data = self._get_json(f"{CIVITAI_API_BASE}/v1/models/{model_id}")
        versions = data.get("modelVersions") or []
        if not versions:
            raise AssetDownloadError(f"Civitai model {model_id} has no versions.")
        return str(versions[0]["id"])

    def _version_info(self, version_id: str):
        data = self._get_json(f"{CIVITAI_API_BASE}/v1/model-versions/{version_id}")
        files = data.get("files") or []
        if not files:
            raise AssetDownloadError(f"Civitai model version {version_id} has no files.")
        file_info = next((f for f in files if f.get("primary")), files[0])

        sha256 = (file_info.get("hashes") or {}).get("SHA256")
        size_kb = file_info.get("sizeKB")
        return {
            "download_url": file_info.get("downloadUrl")
            or f"{CIVITAI_API_BASE}/download/models/{version_id}",
            "filename": file_info.get("name"),
            "size_bytes": int(size_kb * 1024) if size_kb else None,
            "sha256": sha256.lower() if sha256 else None,
        }


class _HuggingFaceSource:
    """Resolves a HuggingFace URL into concrete download metadata."""

    MODEL_FILE_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")

    def __init__(self, token: Optional[str]):
        self.token = token

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def resolve(self, url: str, filename_hint: Optional[str]):
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]

        if "resolve" in parts or "blob" in parts:
            marker = "resolve" if "resolve" in parts else "blob"
            idx = parts.index(marker)
            repo_id = "/".join(parts[:idx])
            revision = parts[idx + 1] if len(parts) > idx + 1 else "main"
            file_path = "/".join(parts[idx + 2:])
            if not file_path:
                raise AssetDownloadError(f"Could not find a file path in URL: {url}")
            download_url = f"{HF_BASE}/{repo_id}/resolve/{revision}/{file_path}"
        else:
            if len(parts) < 2:
                raise AssetDownloadError(f"Could not determine a HuggingFace repo from URL: {url}")
            repo_id = "/".join(parts[:2])
            file_path = filename_hint or self._pick_repo_file(repo_id)
            download_url = f"{HF_BASE}/{repo_id}/resolve/main/{file_path}"

        size_bytes = self._remote_size(download_url)
        return {
            "download_url": download_url,
            "filename": os.path.basename(file_path),
            "size_bytes": size_bytes,
            "sha256": None,
        }

    def _pick_repo_file(self, repo_id: str) -> str:
        try:
            resp = requests.get(f"{HF_BASE}/api/models/{repo_id}", headers=self._headers(), timeout=30)
        except requests.RequestException as e:
            raise AssetDownloadError(f"Failed to reach HuggingFace API for {repo_id}: {e}")
        if resp.status_code == 401:
            raise AssetDownloadError(
                f"HuggingFace returned 401 Unauthorized for {repo_id}. "
                f"Set the {HF_TOKEN_ENV} environment variable."
            )
        resp.raise_for_status()
        siblings = resp.json().get("siblings") or []
        candidates = [
            s["rfilename"] for s in siblings
            if s.get("rfilename", "").lower().endswith(self.MODEL_FILE_EXTS)
        ]
        if not candidates:
            raise AssetDownloadError(
                f"No model file found in {repo_id}. Specify the exact file via "
                "the 'filename' input, or use a full .../resolve/main/<file> URL."
            )
        if len(candidates) > 1:
            raise AssetDownloadError(
                f"Repo {repo_id} has multiple candidate files: {candidates}. "
                "Specify one via the 'filename' input."
            )
        return candidates[0]

    def _remote_size(self, download_url: str):
        try:
            resp = requests.head(download_url, headers=self._headers(), allow_redirects=True, timeout=30)
        except requests.RequestException:
            return None
        if resp.status_code == 401:
            raise AssetDownloadError(
                f"HuggingFace returned 401 Unauthorized for {download_url}. "
                f"Set the {HF_TOKEN_ENV} environment variable."
            )
        size = resp.headers.get("x-linked-size") or resp.headers.get("content-length")
        return int(size) if size else None


def _supports_range(url: str, headers: dict) -> bool:
    try:
        resp = requests.head(url, headers=headers, allow_redirects=True, timeout=30)
    except requests.RequestException:
        return False
    if resp.status_code >= 400:
        return False
    return resp.headers.get("accept-ranges", "").strip().lower() == "bytes"


def _sequential_download(url: str, headers: dict, dest_path: str, total: Optional[int]):
    try:
        resp_ctx = requests.get(url, headers=headers, stream=True, timeout=60)
    except requests.RequestException as e:
        raise AssetDownloadError(f"Failed to start download from {url}: {e}")

    with resp_ctx as resp:
        if resp.status_code == 401:
            raise AssetDownloadError(f"Download request to {url} returned 401 Unauthorized.")
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise AssetDownloadError(f"Download request to {url} failed: {e}")

        total = total or (int(resp.headers["content-length"]) if "content-length" in resp.headers else None)
        progress = ProgressBar(100) if ProgressBar and total else None
        downloaded = 0
        last_log = time.time()

        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(dest_path), prefix=".download_")
        try:
            with os.fdopen(fd, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress:
                        try:
                            progress.update_absolute(int(downloaded / total * 100))
                        except Exception:
                            pass
                    now = time.time()
                    if now - last_log > 2:
                        pct = f"{downloaded / total * 100:.1f}%" if total else f"{downloaded} bytes"
                        print(f"[AssetDownloader] downloading {os.path.basename(dest_path)}: {pct}")
                        last_log = now
            os.replace(tmp_path, dest_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


def _parallel_download(url: str, headers: dict, dest_path: str, total: int, num_workers: int):
    num_workers = max(1, min(num_workers, MAX_ALLOWED_CONNECTIONS))
    part_size = total // num_workers
    ranges = []
    start = 0
    for i in range(num_workers):
        end = total - 1 if i == num_workers - 1 else start + part_size - 1
        ranges.append((start, end))
        start = end + 1

    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(dest_path), prefix=".download_")
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.truncate(total)

    write_lock = threading.Lock()
    progress_state = {"downloaded": 0}
    out_file = open(tmp_path, "r+b")

    def fetch_range(range_start: int, range_end: int):
        pos = range_start
        attempts = 0
        while pos <= range_end:
            attempts += 1
            try:
                range_headers = dict(headers)
                range_headers["Range"] = f"bytes={pos}-{range_end}"
                with requests.get(url, headers=range_headers, stream=True, timeout=60) as resp:
                    if resp.status_code != 206:
                        raise AssetDownloadError(f"range request not honored (status {resp.status_code})")
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        with write_lock:
                            out_file.seek(pos)
                            out_file.write(chunk)
                            progress_state["downloaded"] += len(chunk)
                        pos += len(chunk)
            except (requests.RequestException, AssetDownloadError) as e:
                if attempts >= 3:
                    raise AssetDownloadError(f"Failed to download byte range {range_start}-{range_end}: {e}")
                time.sleep(1.5 * attempts)

    progress = ProgressBar(100) if ProgressBar else None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(fetch_range, s, e) for s, e in ranges]
            last_log = time.time()
            while True:
                _, not_done = concurrent.futures.wait(futures, timeout=2)
                pct = progress_state["downloaded"] / total * 100
                if progress:
                    try:
                        progress.update_absolute(int(pct))
                    except Exception:
                        pass
                now = time.time()
                if now - last_log > 2:
                    print(
                        f"[AssetDownloader] downloading ({num_workers} connections) "
                        f"{os.path.basename(dest_path)}: {pct:.1f}%"
                    )
                    last_log = now
                if not not_done:
                    break
            for fut in futures:
                fut.result()
    except BaseException:
        out_file.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    out_file.close()
    os.replace(tmp_path, dest_path)


def _download_asset(
    url: str,
    headers: dict,
    dest_path: str,
    expected_size: Optional[int] = None,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
):
    total = expected_size
    if total is None:
        try:
            head = requests.head(url, headers=headers, allow_redirects=True, timeout=30)
            if "content-length" in head.headers:
                total = int(head.headers["content-length"])
        except requests.RequestException:
            total = None

    eligible_for_parallel = (
        max_connections > 1 and total is not None and total >= MIN_SIZE_FOR_PARALLEL
    )
    if eligible_for_parallel and _supports_range(url, headers):
        try:
            _parallel_download(url, headers, dest_path, total, max_connections)
            return
        except Exception as e:
            print(f"[AssetDownloader] parallel download failed ({e}), falling back to a single connection.")

    _sequential_download(url, headers, dest_path, total)


class ModelAssetDownloader:
    CATEGORY = "loaders/downloaders"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "download"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": ""}),
                "save_type": (SAVE_TYPES, {"default": "checkpoints"}),
            },
            "optional": {
                "filename": ("STRING", {"default": ""}),
                "overwrite": ("BOOLEAN", {"default": False}),
                "max_connections": ("INT", {"default": DEFAULT_MAX_CONNECTIONS, "min": 1, "max": MAX_ALLOWED_CONNECTIONS}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, url, save_type, filename="", overwrite=False, max_connections=DEFAULT_MAX_CONNECTIONS):
        # Always re-check: the node itself decides cheaply whether a real
        # download is needed by comparing the local file against remote
        # metadata (hash/size) before doing any heavy network transfer.
        return float("nan")

    def download(self, url, save_type, filename="", overwrite=False, max_connections=DEFAULT_MAX_CONNECTIONS):
        url = (url or "").strip()
        if not url:
            raise AssetDownloadError("url is empty.")

        host = urlparse(url).netloc.lower()
        save_dir = _get_save_dir(save_type)

        if host == "civitai.com" or host.endswith(".civitai.com"):
            token = os.environ.get(CIVITAI_TOKEN_ENV)
            info = _CivitaiSource(token).resolve(url)
        elif host == "huggingface.co" or host.endswith(".huggingface.co"):
            token = os.environ.get(HF_TOKEN_ENV)
            info = _HuggingFaceSource(token).resolve(url, filename.strip() or None)
        else:
            raise AssetDownloadError(
                f"Unsupported host '{host}'. Only huggingface.co and civitai.com URLs are supported."
            )

        headers = {"Authorization": f"Bearer {token}"} if token else {}
        final_name = _sanitize_filename(filename.strip() or info["filename"] or "")
        dest_path = os.path.join(save_dir, final_name)

        if os.path.exists(dest_path) and not overwrite:
            if self._matches_existing(dest_path, info):
                print(f"[AssetDownloader] '{final_name}' already exists and matches the remote file, skipping.")
                return (dest_path,)
            print(f"[AssetDownloader] '{final_name}' exists but does not match the remote file, re-downloading.")

        print(f"[AssetDownloader] downloading {info['download_url']} -> {dest_path}")
        _download_asset(info["download_url"], headers, dest_path, info.get("size_bytes"), max_connections)
        return (dest_path,)

    @staticmethod
    def _matches_existing(dest_path: str, info: dict) -> bool:
        expected_sha256 = info.get("sha256")
        expected_size = info.get("size_bytes")

        if expected_sha256:
            stat = os.stat(dest_path)
            cached = _load_hash_cache(dest_path)
            if (
                cached
                and cached.get("size") == stat.st_size
                and cached.get("mtime") == stat.st_mtime
                and cached.get("sha256") == expected_sha256
            ):
                # File hasn't been touched since we last verified it against
                # this exact remote hash, so skip re-reading the whole thing.
                return True

            actual_sha256 = _sha256_file(dest_path)
            if actual_sha256 == expected_sha256:
                _save_hash_cache(dest_path, stat.st_size, stat.st_mtime, actual_sha256)
                return True
            return False

        if expected_size:
            return os.path.getsize(dest_path) == expected_size

        # Neither a hash nor a size was available to compare against: don't
        # assume a same-named local file is the right one, re-download it.
        return False


NODE_CLASS_MAPPINGS = {"ModelAssetDownloader": ModelAssetDownloader}
NODE_DISPLAY_NAME_MAPPINGS = {"ModelAssetDownloader": "Download Model/LoRA (HF/Civitai)"}
