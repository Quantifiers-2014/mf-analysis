"""Parsers and fetchers for external data sources."""


class SourceError(ValueError):
    """Raised when a source file or response cannot be parsed or fails a hard check."""
