import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class Event:
    time: float
    priority: int
    sequence: int
    kind: str = field(compare=False)
    payload: dict = field(compare=False, default_factory=dict)


class EventQueue:
    def __init__(self):
        self.heap = []
        self.sequence = 0
        self.clock = 0.0

    def push(self, time, kind, payload, priority=1):
        if time < self.clock:
            raise ValueError("Cannot schedule an event in the past")
        heapq.heappush(self.heap, Event(time, priority, self.sequence, kind, payload))
        self.sequence += 1

    def drain(self):
        while self.heap:
            event = heapq.heappop(self.heap)
            self.clock = event.time
            yield event
