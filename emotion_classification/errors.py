"""Exceptions shared across the package."""


class ConfigError(ValueError):
    """A run names a corpus, schema, split, evaluation set or model that does not exist.

    Raised before any training starts. It subclasses ``ValueError`` so library
    callers can catch either; the command line turns it into a one-line message.
    """
