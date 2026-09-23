import pytest

from kresearch.fetch.web import UnsafeURLError, check_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
        "http://10.0.0.5/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://[::1]/",
        "ftp://example.com/",
        "file:///etc/passwd",
    ],
)
def test_check_url_blocks_unsafe_targets(url):
    with pytest.raises(UnsafeURLError):
        check_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/",
        "http://example.com/some/path",
    ],
)
def test_check_url_allows_public_targets(url):
    check_url(url)  # should not raise


def test_check_url_rejects_unresolvable_hostname():
    with pytest.raises(UnsafeURLError):
        check_url("http://this-host-should-not-resolve.invalid/")
