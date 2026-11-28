# Copyright (c) ModelScope Contributors. All rights reserved.
"""The CPU worker a parallel streaming transform fans its keyspace partitions out to."""
from twinkle import remote_class, remote_function

#: The CPU DeviceGroup the partition workers occupy; one rank per partition.
PARTITION_GROUP = 'weight_transform'


@remote_class()
class PartitionWorker:
    """Run keyspace partitions of a streaming transform; under ray each rank takes its slice of jobs."""

    @remote_function(dispatch='slice', collect='flatten')
    def __call__(self, jobs):
        """``jobs`` is a list of ``(transform, partition_keys, rank, world_size, output_dir)``."""
        return [
            transform.run_partition(keys, rank, world_size, output_dir)
            for transform, keys, rank, world_size, output_dir in jobs
        ]
