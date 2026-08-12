from obsidianlink.env.capabilities import (
    BackendCapabilities,
    CAPABILITY_IDS,
    CapabilityMismatchError,
    assert_backend_can_start_task,
    assert_casting_c1_capabilities,
    missing_for_casting_c1,
)
__all__ = [
    "BackendCapabilities",
    "CAPABILITY_IDS",
    "CapabilityMismatchError",
    "FakeEnvironmentBackend",
    "assert_backend_can_start_task",
    "assert_casting_c1_capabilities",
    "missing_for_casting_c1",
]


def __getattr__(name: str):
    """Keep the legacy FakeBackend import lazy for clean v2 kernel imports."""

    if name == "FakeEnvironmentBackend":
        from obsidianlink.env.fake import FakeEnvironmentBackend

        return FakeEnvironmentBackend
    raise AttributeError(name)
