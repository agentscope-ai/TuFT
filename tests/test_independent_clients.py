"""Timing contract for the static multi-tenant GPU workload; CPU only."""

import asyncio

import pytest
from independent_clients import IndependentClients


def test_nested_tenant_sets_keep_independent_arrivals_and_delays():
    clients = IndependentClients(20261106)
    eight = clients.metadata(8)["initial_delays_s"]
    assert clients.metadata(4)["initial_delays_s"] == eight[:4]
    assert len(set(eight)) == 8
    assert all(0 <= x < 240 for x in eight)
    original = clients.tool_delay(2, 3, 4, 1)
    for i in reversed(range(100)):
        clients.tool_delay(i, i, i, 2)
    assert clients.tool_delay(2, 3, 4, 1) == original
    assert IndependentClients(123).start_delay(2) != clients.start_delay(2)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(observation_s=0),
        dict(arrival_window_s=-1),
        dict(think_max_s=float("inf")),
        dict(tool_delay_max_s=float("nan")),
    ],
)
def test_invalid_windows(kwargs):
    with pytest.raises(ValueError):
        IndependentClients(1, **kwargs)


def test_round_admission_stops_at_boundary():
    clients = IndependentClients(1, arrival_window_s=10, observation_s=20)
    assert clients.can_start_round(29.99)
    assert not clients.can_start_round(30)
    assert not clients.can_start_round(31)
    assert clients.think_delay(0, 0) == 0


def test_async_clients_have_no_arrival_barrier():
    clients = IndependentClients(42, arrival_window_s=0.04, observation_s=0.1)

    async def run():
        events = []

        async def client(i):
            await asyncio.sleep(clients.start_delay(i))
            events.append(i)

        await asyncio.gather(*(client(i) for i in range(4)))
        return events

    assert asyncio.run(run()) == sorted(range(4), key=clients.start_delay)
