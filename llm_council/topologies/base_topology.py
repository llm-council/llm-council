# Global registry to store topologies
TOPOLOGY_REGISTRY = {}


def topology(topology_name: str):
    """Class decorator to register topologies with metadata."""

    def decorator(cls):
        # Store class reference and metadata in the registry
        TOPOLOGY_REGISTRY[topology_name] = {
            "class": cls,
        }

        return cls  # Return the class unchanged

    return decorator
