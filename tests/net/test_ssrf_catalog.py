"""The SSRF test catalog of the egress guard, part 1: addresses, URLs and name resolution (ADR 0017).

Every vector must be refused. Part 2 (``test_http_guard.py``) runs the vectors that need an HTTP exchange:
redirects, rebinding between hops, ports on redirects, size, encoding and the deadline.
"""

from __future__ import annotations

import asyncio
import ipaddress

import pytest

from orkexa.net.guard import (
    METADATA_HOSTS,
    EgressError,
    EgressPolicy,
    GuardedBackend,
    address_problem,
    redact_url,
    url_secret_problem,
)

# ---- addresses ---------------------------------------------------------------------------------------------

#: address -> reason. The first block is the vector list of nodestro-runtime's MCP guard, unchanged (parity).
REFUSED_ADDRESSES = {
    "127.0.0.1": "loopback",
    "127.255.0.9": "loopback",
    "::1": "loopback",
    "10.0.0.1": "private",
    "172.16.5.4": "private",
    "192.168.1.1": "private",
    "169.254.169.254": "metadata",
    "169.254.170.2": "metadata",
    "fd00:ec2::254": "metadata",
    "100.100.100.200": "metadata",
    "169.254.10.10": "link_local",
    "fe80::1": "link_local",
    "fc00::1": "private",
    "fec0::1": "site_local",
    "100.64.0.1": "shared",
    "0.0.0.0": "unspecified",
    "::": "unspecified",
    "224.0.0.1": "multicast",
    "ff02::1": "multicast",
    "::ffff:127.0.0.1": "loopback",
    "::ffff:10.1.2.3": "private",
    "64:ff9b::7f00:1": "loopback",
    "64:ff9b::a9fe:a9fe": "metadata",
    "2002:7f00:1::": "loopback",
    "::127.0.0.1": "loopback",
    "192.0.2.1": "private",
    "240.0.0.1": "private",
    # Added for NAL-A2.
    "0.1.2.3": "private",  # 0.0.0.0/8 ("this network"; Linux connects 0.x to the host itself)
    "100.127.255.254": "shared",
    "172.31.255.255": "private",
    "fd12:3456::1": "private",
    "febf::1": "link_local",
    "ff05::1:3": "multicast",
    "239.255.255.250": "multicast",
    "255.255.255.255": "private",
    "198.18.0.1": "private",
    "2001:0:4136:e378:8000:63bf:3fff:fdd2": "private",  # Teredo, whatever client it embeds
    "2002:a9fe:a9fe::": "metadata",  # 6to4 around the metadata address
    "2002:0a00:0001::1": "private",  # 6to4 around 10.0.0.1
    "64:ff9b:1::a00:1": "private",  # local-use NAT64 around 10.0.0.1
    "::ffff:169.254.169.254": "metadata",
    "::ffff:100.64.0.1": "shared",
    "2001:db8::1": "private",  # documentation
}


@pytest.mark.parametrize(("value", "reason"), sorted(REFUSED_ADDRESSES.items()))
def test_internal_addresses_are_refused(value: str, reason: str) -> None:
    assert address_problem(ipaddress.ip_address(value)) == reason


@pytest.mark.parametrize(
    "value", ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111", "::ffff:8.8.8.8"]
)
def test_global_addresses_pass(value: str) -> None:
    assert address_problem(ipaddress.ip_address(value)) is None


def test_scoped_addresses_are_refused() -> None:
    assert address_problem(ipaddress.ip_address("fe80::1%eth0")) == "scoped"


def test_loopback_consent_allows_only_loopback() -> None:
    assert address_problem(ipaddress.ip_address("127.0.0.1"), allow_loopback=True) is None
    assert address_problem(ipaddress.ip_address("::1"), allow_loopback=True) is None
    assert address_problem(ipaddress.ip_address("10.0.0.1"), allow_loopback=True) == "private"
    assert address_problem(ipaddress.ip_address("169.254.169.254"), allow_loopback=True) == "metadata"


@pytest.mark.parametrize(
    "value", ["127.0.0.1", "10.1.2.3", "172.20.0.5", "192.168.0.10", "100.64.1.1", "fd00::5"]
)
def test_allow_private_opens_the_private_ranges(value: str) -> None:
    assert address_problem(ipaddress.ip_address(value), allow_private=True) is None


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("169.254.169.254", "metadata"),
        ("100.100.100.200", "metadata"),  # inside 100.64.0.0/10, still refused
        ("fd00:ec2::254", "metadata"),  # inside fc00::/7, still refused
        ("169.254.1.1", "link_local"),
        ("fe80::1", "link_local"),
        ("0.0.0.0", "unspecified"),
        ("0.1.2.3", "private"),  # 0.0.0.0/8 is not a private network
        ("224.0.0.1", "multicast"),
        ("192.0.2.1", "private"),
        ("240.0.0.1", "private"),
        ("2001:0:4136:e378:8000:63bf:3fff:fdd2", "private"),
    ],
)
def test_allow_private_keeps_the_rest_refused(value: str, reason: str) -> None:
    assert address_problem(ipaddress.ip_address(value), allow_private=True) == reason


# ---- URLs --------------------------------------------------------------------------------------------------


def refusal(policy: EgressPolicy, url: str) -> tuple[str, str | None]:
    with pytest.raises(EgressError) as caught:
        policy.check_url(url)
    return caught.value.code, caught.value.reason


#: url -> (code, reason) with the default policy.
REFUSED_URLS = {
    # numeric spellings of 127.0.0.1 that getaddrinfo and browsers accept
    "http://2130706433/": ("address_refused", "numeric_host"),
    "http://0x7f000001/": ("address_refused", "numeric_host"),
    "http://127.1/": ("address_refused", "numeric_host"),
    "http://0177.0.0.1/": ("address_refused", "numeric_host"),
    "http://0x7f.0.0.1/": ("address_refused", "numeric_host"),
    "http://017700000001/": ("address_refused", "numeric_host"),
    "http://127.0.0.1./": ("address_refused", "loopback"),
    "http://3232235777/": ("address_refused", "numeric_host"),  # 192.168.1.1
    # literals
    "http://127.0.0.1/": ("address_refused", "loopback"),
    "http://[::1]/": ("address_refused", "loopback"),
    "http://[::ffff:127.0.0.1]/": ("address_refused", "loopback"),
    "http://[::ffff:7f00:1]/": ("address_refused", "loopback"),
    "http://[0:0:0:0:0:ffff:a9fe:a9fe]/": ("address_refused", "metadata"),
    "http://169.254.169.254/latest/meta-data/": ("address_refused", "metadata"),
    "http://[fd00:ec2::254]/": ("address_refused", "metadata"),
    "http://10.0.0.1:443/": ("address_refused", "private"),
    "http://0.0.0.0/": ("address_refused", "unspecified"),
    "http://[fe80::1%25eth0]/": ("address_refused", "scoped"),
    # names
    "http://localhost/": ("address_refused", "loopback"),
    "http://LOCALHOST./": ("address_refused", "loopback"),
    "http://api.localhost/": ("address_refused", "loopback"),
    "http://metadata.google.internal/computeMetadata/v1/": ("address_refused", "metadata"),
    "http://metadata/": ("address_refused", "metadata"),
    "http://INSTANCE-DATA./latest": ("address_refused", "metadata"),
    "http://metadata.azure.com/": ("address_refused", "metadata"),
    # syntax, scheme, port, credentials
    "http://exa%6dple.com/": ("url_invalid", None),
    "http://exa mple.com/": ("url_invalid", None),
    "http://[not-an-ip]/": ("url_invalid", None),
    "file:///etc/passwd": ("url_invalid", None),
    "gopher://example.com/": ("url_invalid", None),
    "ftp://example.com/": ("url_invalid", None),
    "//example.com/": ("url_invalid", None),
    "http://user:pw@example.com/": ("url_credentials", None),
    "http://user@example.com/": ("url_credentials", None),
    "http://example.com:22/": ("port_refused", None),
    "http://example.com:6379/": ("port_refused", None),
    "https://example.com:8443/": ("port_refused", None),
    "http://example.com:99999/": ("url_invalid", None),
}


@pytest.mark.parametrize(("url", "expected"), sorted(REFUSED_URLS.items()))
def test_refused_urls(url: str, expected: tuple[str, str | None]) -> None:
    assert refusal(EgressPolicy(), url) == expected


def test_the_catalog_has_at_least_forty_vectors() -> None:
    assert len(REFUSED_ADDRESSES) + len(REFUSED_URLS) >= 40


def test_allowed_urls_return_their_origin() -> None:
    policy = EgressPolicy()
    assert policy.check_url("https://Example.COM./a?b=c") == ("https", "example.com", 443)
    assert policy.check_url("http://example.com/") == ("http", "example.com", 80)
    assert policy.check_url("https://93.184.216.34/") == ("https", "93.184.216.34", 443)
    assert policy.check_url("https://[2606:4700:4700::1111]/") == ("https", "2606:4700:4700::1111", 443)
    assert policy.check_url("https://bücher.example/") == ("https", "xn--bcher-kva.example", 443)
    assert policy.check_url("https://web-1.example/") == ("https", "web-1.example", 443)


def test_schemes_ports_and_lists() -> None:
    https_only = EgressPolicy(schemes=frozenset({"https"}))
    assert refusal(https_only, "http://example.com/") == ("scheme_refused", None)
    any_port = EgressPolicy(ports=None)
    assert any_port.check_url("https://example.com:8443/").port == 8443
    listed = EgressPolicy(allow_hosts=frozenset({"api.example.com", "*.search.example"}))
    assert listed.check_url("https://api.example.com/")
    assert listed.check_url("https://eu.search.example/")
    assert refusal(listed, "https://search.example/") == ("host_refused", "not_allowed")
    assert refusal(listed, "https://other.example.com/") == ("host_refused", "not_allowed")
    assert refusal(listed, "https://api.example.com.evil.test/") == ("host_refused", "not_allowed")
    denied = EgressPolicy(deny_hosts=frozenset({"*.tracker.example", "bad.example"}))
    assert refusal(denied, "https://x.tracker.example/") == ("host_refused", "denied")
    assert refusal(denied, "https://tracker.example/") == ("host_refused", "denied")
    assert refusal(denied, "https://BAD.example./") == ("host_refused", "denied")
    assert denied.check_url("https://good.example/")


def test_an_allow_list_never_allows_a_refused_address() -> None:
    policy = EgressPolicy(allow_hosts=frozenset({"10.0.0.5", "localhost", "metadata.google.internal"}))
    assert refusal(policy, "http://10.0.0.5/") == ("address_refused", "private")
    assert refusal(policy, "http://localhost/") == ("address_refused", "loopback")
    assert refusal(policy, "http://metadata.google.internal/") == ("address_refused", "metadata")


def test_allow_private_for_a_self_hosted_service() -> None:
    policy = EgressPolicy(
        allow_private=True, allow_hosts=frozenset({"searx.lan", "192.168.1.20", "localhost"})
    )
    assert policy.check_url("http://searx.lan/search")
    assert policy.check_url("http://192.168.1.20/search")
    assert policy.check_url("http://localhost/search")
    assert refusal(policy, "http://169.254.169.254/") == ("address_refused", "metadata")
    assert refusal(policy, "http://metadata/") == ("address_refused", "metadata")
    assert refusal(policy, "http://2130706433/") == ("address_refused", "numeric_host")


def test_metadata_names_are_refused_by_every_policy() -> None:
    assert "metadata.google.internal" in METADATA_HOSTS
    policy = EgressPolicy(deny_hosts=frozenset({"other.example"}), allow_private=True)
    for host in METADATA_HOSTS:
        assert refusal(policy, f"http://{host}/") == ("address_refused", "metadata")


def test_invalid_policies_are_refused() -> None:
    with pytest.raises(ValueError):
        EgressPolicy(schemes=frozenset({"ftp"}))
    with pytest.raises(ValueError):
        EgressPolicy(max_bytes=0)
    with pytest.raises(ValueError):
        EgressPolicy(timeout=0)
    with pytest.raises(ValueError):
        EgressPolicy(ports=frozenset({0}))
    with pytest.raises(ValueError):
        EgressPolicy(allow_hosts=frozenset({"bad host"}))


# ---- name resolution: DNS pinning --------------------------------------------------------------------------


def resolver_from(answers: dict[str, list[list[str]]]):  # type: ignore[no-untyped-def]
    """A resolver whose answers change per lookup (rebinding)."""
    calls: dict[str, int] = {}

    async def resolve(host: str, port: int) -> list[str]:
        index = calls.get(host, 0)
        calls[host] = index + 1
        rounds = answers[host]
        return rounds[min(index, len(rounds) - 1)]

    return resolve


def backend_refusal(backend: GuardedBackend, host: str) -> tuple[str, str | None]:
    with pytest.raises(EgressError) as caught:
        asyncio.run(backend.resolve(host, 443))
    return caught.value.code, caught.value.reason


def test_every_answer_is_checked_and_rebinding_is_caught() -> None:
    backend = GuardedBackend(
        resolver=resolver_from(
            {
                "public.test": [["93.184.216.34"]],
                "mixed.test": [["93.184.216.34", "10.0.0.1"]],
                "rebind.test": [["93.184.216.34"], ["127.0.0.1"]],
                "v6.test": [["::ffff:169.254.169.254"]],
                "nat64.test": [["2606:4700:4700::1111", "64:ff9b::a00:1"]],
                "scoped.test": [["fe80::1%eth0"]],
                "empty.test": [[]],
                "garbage.test": [["not-an-address"]],
            }
        )
    )
    assert asyncio.run(backend.resolve("public.test", 443)) == [ipaddress.ip_address("93.184.216.34")]
    assert backend_refusal(backend, "mixed.test") == ("address_refused", "private")
    assert backend_refusal(backend, "v6.test") == ("address_refused", "metadata")
    assert backend_refusal(backend, "nat64.test") == ("address_refused", "private")
    assert backend_refusal(backend, "scoped.test") == ("address_refused", "scoped")
    assert backend_refusal(backend, "empty.test") == ("dns_failed", None)
    assert backend_refusal(backend, "garbage.test") == ("dns_failed", None)
    # The first connection gets the public answer, the next one (pool refill, redirect) the rebound one.
    assert asyncio.run(backend.resolve("rebind.test", 443))
    assert backend_refusal(backend, "rebind.test") == ("address_refused", "loopback")


def test_the_backend_refuses_metadata_names_and_literals_without_a_lookup() -> None:
    async def never(host: str, port: int) -> list[str]:
        raise AssertionError("no lookup expected")

    backend = GuardedBackend(resolver=never)
    assert backend_refusal(backend, "metadata.google.internal.") == ("address_refused", "metadata")
    assert backend_refusal(backend, "169.254.169.254") == ("address_refused", "metadata")
    assert backend_refusal(backend, "[::1]") == ("address_refused", "loopback")


def test_numeric_host_forms_resolve_to_refused_addresses_at_the_backend_too() -> None:
    # The URL check refuses them first; should one reach the backend, the system's answer is checked as well.
    backend = GuardedBackend(
        resolver=resolver_from({"2130706433": [["127.0.0.1"]], "0x7f000001": [["127.0.0.1"]]})
    )
    assert backend_refusal(backend, "2130706433") == ("address_refused", "loopback")
    assert backend_refusal(backend, "0x7f000001") == ("address_refused", "loopback")


def test_dns_errors_and_unix_sockets() -> None:
    async def failing(host: str, port: int) -> list[str]:
        raise OSError("no such name")

    backend = GuardedBackend(resolver=failing)
    assert backend_refusal(backend, "missing.test") == ("dns_failed", None)
    with pytest.raises(EgressError) as caught:
        asyncio.run(backend.connect_unix_socket("/var/run/docker.sock"))
    assert caught.value.reason == "unix"


def test_allow_private_reaches_the_backend() -> None:
    backend = GuardedBackend(
        allow_private=True, allow_loopback=True, resolver=resolver_from({"searx.lan": [["192.168.1.20"]]})
    )
    assert asyncio.run(backend.resolve("searx.lan", 80)) == [ipaddress.ip_address("192.168.1.20")]
    metadata = GuardedBackend(allow_private=True, resolver=resolver_from({"x.lan": [["169.254.169.254"]]}))
    assert backend_refusal(metadata, "x.lan") == ("address_refused", "metadata")


def test_the_error_factory_can_be_replaced() -> None:
    class HostError(Exception):
        def __init__(self, code: str, message: str, details: dict[str, object]) -> None:
            super().__init__(message)
            self.code = code
            self.details = details

    class HostBackend(GuardedBackend):
        def error(self, code: str, message: str, **details: object) -> Exception:
            return HostError(f"host.{code}", message, details)

    backend = HostBackend(resolver=resolver_from({"internal.test": [["10.0.0.1"]]}))
    with pytest.raises(HostError) as caught:
        asyncio.run(backend.resolve("internal.test", 443))
    assert caught.value.code == "host.address_refused"
    assert str(caught.value) == "internal.test resolves to a refused address (private)"


# ---- URLs in messages --------------------------------------------------------------------------------------


def test_messages_never_show_queries_or_credentials() -> None:
    assert (
        redact_url("https://user:pw@api.example.com/v1/search?api_key=SECRET#x")
        == "https://api.example.com/v1/search?…"
    )
    assert redact_url("https://[::1]:8080/a") == "https://[::1]:8080/a"
    assert redact_url("http://example.com:99999/") == "(invalid address)"
    with pytest.raises(EgressError) as caught:
        EgressPolicy().check_url("http://10.0.0.1/?token=SECRET")
    assert "SECRET" not in str(caught.value)
    assert url_secret_problem("https://example.com/?api_key=x") == "query"
    assert url_secret_problem("https://example.com/#frag") == "fragment"
    assert url_secret_problem("https://example.com/page") is None
