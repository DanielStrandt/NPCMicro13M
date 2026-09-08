# Security policy

## Supported versions

Security fixes are currently maintained for the latest published release.

## Reporting a vulnerability

Please do not publish credentials, private transcripts, or exploitable
details in a public issue. Contact the repository owner privately through the
contact method listed on the GitHub profile, including the affected file or
endpoint, reproduction steps, and a minimal proof of impact.

The local API binds to `127.0.0.1` by default. Do not expose it to a network
without adding authentication, access control, rate limiting, and an
appropriate reverse proxy.
