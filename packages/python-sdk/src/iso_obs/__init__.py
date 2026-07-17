"""iso-obs: the Reliability Studio Python SDK.

The SDK instruments evaluation runs of autonomous systems and streams their
traces to the Reliability Studio API. Typical usage pairs
:class:`ReliabilityClient` (authentication, transport, resources) with
:class:`iso_obs.run.RunContext` (run lifecycle and event batching):

    >>> from iso_obs import ReliabilityClient
    >>> from iso_obs.run import RunContext

See :mod:`iso_obs.metrics` for custom metric declaration and
:mod:`iso_obs.environments` for the simulator adapter interface.
"""

from __future__ import annotations

from .client import ReliabilityClient
from .run import RunContext

__version__ = "0.1.0"

__all__ = ["ReliabilityClient", "RunContext", "__version__"]
