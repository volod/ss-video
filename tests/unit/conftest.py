"""Session-wide fixtures and import pre-warming for unit tests.

Pre-imports torchvision at session start to avoid a known circular-import bug
(torchvision._meta_registrations tries to access torchvision.extension before
the package is fully initialised when first imported inside a sys.modules
patch.dict context).
"""


def pytest_configure(config):
    """Pre-warm torchvision so it is fully initialised before any test runs."""
    try:
        import torchvision  # noqa: F401
    except Exception:
        pass
