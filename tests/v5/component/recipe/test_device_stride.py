"""defect80 (5feb7e9eb) / 13-D-1: the device plan's nproc_per_node is the per-node device stride, not the
global rank span, and it is read from the Ray CLUSTER (ResourceManager.infer_nproc_per_node), not from
the driver's local device_count. ResourceManager maps rank->(node,device) via rank//stride /
rank%stride and filters nodes by device_count >= stride; passing the span made every plan whose
disjoint auxiliary ranks overflow the visible devices fail with a misleading "available: 0".

PLAN (case | input | expected | kills):
 1 span-fits | span3 nodes[4] | 3 | always-per_node over-reservation
 2 span-equals | span4 nodes[4] | 4 | off-by-one (< instead of <=)
 3 span-overflows | span5 nodes[4] | 4 | old stride=span bug (defect80 itself)
 4 span1 | span1 nodes[8] | 1 | single-GPU minimal reservation lost
 5 heterogeneous | span9 nodes[8,4] | 8 | min/max mix-up: a 4-device stride maps ranks 8 onto node 2
 6 dead-node | span5 nodes[8 dead, 4 alive] | 4 | dead node counted
 7 no-device-node | nodes[{}] | ValueError | silent stride 0

Oracle = the ResourceManager contract (rank//stride node mapping), hand-computed min().
Reverse-verify: revert min() to `return rank_span` -> case 3 RED (5 != 4), cases 1/2/4 stay GREEN.
"""

import pytest

from twinkle.infra._ray.resource_manager import ResourceManager


def _nodes(*counts, dead=()):
    return [{'Alive': index not in dead, 'Resources': {'GPU': count}} for index, count in enumerate(counts)]


@pytest.mark.parametrize('span,nodes,expected', [
    (3, _nodes(4), 3),        # span-fits: minimal reservation, all ranks on node 0
    (4, _nodes(4), 4),        # span-equals
    (5, _nodes(4), 4),        # span-overflows: honest per-node stride, not the misleading span
    (1, _nodes(8), 1),        # span1
    (9, _nodes(8, 4), 8),     # heterogeneous cluster: the largest per-node count
    (5, _nodes(8, 4, dead=(0, )), 4),  # only alive nodes count
])
def test_stride_is_min_of_span_and_cluster_per_node(span, nodes, expected):
    assert ResourceManager.infer_nproc_per_node('GPU', span, nodes) == expected


def test_no_device_node_fails_loudly():
    with pytest.raises(ValueError, match='no alive node of the Ray cluster reports a GPU'):
        ResourceManager.infer_nproc_per_node('GPU', 3, _nodes(0))

