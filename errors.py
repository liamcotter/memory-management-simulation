class SecurityError(Exception):
    """Raised when a memory request is outside of the permitted ranges."""
    ...

class SimulatedMemoryError(MemoryError):
    """Custom Equivalent to the MemoryError"""
    ...

class InvalidMemoryRequestError(MemoryError):
    """The request is invalid, but it is not a security error."""
    ...

class invalidInitialisationParameterError(Exception):
    """Raised when impossible initialisation parameters are passed."""
    ...

class MissingBlockFrameError(Exception):
    """This error is raised when a block frame method is called but the block is not in a frame."""
    ...

class AllocatedBlockFrameError(Exception):
    """This error is raised when an operation on a block frame is attempted when it is not free."""
    ...