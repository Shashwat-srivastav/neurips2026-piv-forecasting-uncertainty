"""Minimal stand-in for the `requests` package, for the evaluation container
only (confirmed missing 27 Aug via a real import-blocking test, alongside the
already-known-absent scipy/h5py/einops/matplotlib/pandas).

CNO_libs/dnnlib/util.py imports `requests` unconditionally at module level,
but the only functions that call it (`open_url()`, a StyleGAN-style remote
pretrained-weight downloader) are never reached when loading a local
checkpoint for inference -- nothing in this submission's predict() path calls
them. This stub only needs to make `import requests` and the module-level
references (`requests.compat`, `requests.Session`) not raise; it never needs
to make an actual HTTP request.

Usage, at the top of submission.py, BEFORE importing anything that pulls in
rpde_baselines.model.cno:

    import sys
    try:
        import requests            # noqa: F401 -- real one, if present, wins
    except ImportError:
        import _vendor_requests_stub as requests
        sys.modules["requests"] = requests
"""
import urllib.parse as _up


class compat:
    @staticmethod
    def urlparse(url):
        return _up.urlparse(url)

    @staticmethod
    def urljoin(base, url):
        return _up.urljoin(base, url)


class Session:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, *a, **kw):
        raise RuntimeError(
            "requests stub: open_url() was actually called at runtime -- "
            "this should never happen for a local-checkpoint inference path. "
            "If it did, something is trying to download a remote pretrained "
            "weight; investigate before assuming the stub is at fault.")


class exceptions:
    class RequestException(Exception):
        pass
