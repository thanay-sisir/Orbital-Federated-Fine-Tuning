import copy
import math

from .network import LinkEngine


POLICIES = ("connected", "fixed", "contact_aware", "feasible_fair")


def simulate(contacts, network, clients, mission_seconds, round_seconds, payload_bytes, step_seconds, max_steps=4, policy="fixed"):
    if policy not in POLICIES or min(payload_bytes, step_seconds, max_steps, round_seconds) <= 0:
        raise ValueError("Invalid feasibility inputs")
    if policy == "connected":
        contacts = [{"client": i, "station": "connected-ground", "start": 0, "end": mission_seconds, "delay_seconds": 0} for i in range(clients)]
    engine = LinkEngine(contacts, network)
    age, accepted_counts = [0] * clients, [0] * clients
    summaries = []
    round_id, start = 0, 0
    for _ in range(math.ceil(mission_seconds / round_seconds)):
        deadline = min(start + round_seconds, mission_seconds)
        if start >= deadline:
            break
        tasks, uploads, attempted = [], [], 0
        order = list(range(clients))
        if policy == "contact_aware":
            order.sort(key=lambda i: (min((c["end"] for c in contacts if c["client"] == i and c["end"] > start), default=float("inf")), i))
        if policy == "feasible_fair":
            pending = set(order)
            while pending:
                proposals = []
                for client in sorted(pending):
                    for steps in sorted({1, max(1, max_steps // 2), max_steps}, reverse=True):
                        trial = copy.deepcopy(engine)
                        down = trial.transfer(client, payload_bytes, start, deadline, "download", f"r{round_id}-c{client}-down")
                        if not down["complete"]:
                            continue
                        complete = down["arrival"] + steps * step_seconds
                        if complete >= deadline:
                            continue
                        up = trial.transfer(client, payload_bytes, complete, deadline, "upload", f"r{round_id}-c{client}-up")
                        if up["complete"]:
                            proposals.append((-age[client], up["arrival"], client, steps, trial, up))
                            break
                if not proposals:
                    break
                _, _, client, steps, engine, up = min(proposals, key=lambda p: p[:3])
                pending.remove(client)
                attempted += steps
                uploads.append((client, steps, up))
        else:
            for client in order:
                down = engine.transfer(client, payload_bytes, start, deadline, "download", f"r{round_id}-c{client}-down")
                if not down["complete"]:
                    continue
                complete = down["arrival"] + max_steps * step_seconds
                if complete > deadline:
                    continue
                tasks.append((complete, client))
            for complete, client in sorted(tasks):
                attempted += max_steps
                up = engine.transfer(client, payload_bytes, complete, deadline, "upload", f"r{round_id}-c{client}-up")
                if up["complete"]:
                    uploads.append((client, max_steps, up))
        succeeded = len(uploads) >= network.min_participants
        accepted = {client for client, _, _ in uploads} if succeeded else set()
        for client in range(clients):
            accepted_counts[client] += int(client in accepted)
            age[client] = 0 if client in accepted else age[client] + 1
        summaries.append({
            "round": round_id, "start": start, "deadline": deadline,
            "delivered_clients": [c for c, _, _ in uploads], "accepted_clients": sorted(accepted),
            "scheduled_steps": attempted,
            "accepted_steps": sum(steps for _, steps, _ in uploads) if succeeded else 0,
            "successful": succeeded,
            "last_upload_arrival": max((up["arrival"] for _, _, up in uploads), default=None),
        })
        start, round_id = deadline, round_id + 1
    norm = sum(n * n for n in accepted_counts)
    return {
        "policy": policy, "rounds": summaries,
        "successful_rounds": sum(r["successful"] for r in summaries),
        "scheduled_steps": sum(r["scheduled_steps"] for r in summaries),
        "accepted_steps": sum(r["accepted_steps"] for r in summaries),
        "accepted_updates": sum(accepted_counts), "accepted_per_client": accepted_counts,
        "jain_participation_fairness": sum(accepted_counts) ** 2 / (clients * norm) if norm else 0,
        "wire_bytes": engine.wire_bytes,
        "complete_transfers": sum(r["complete"] for r in engine.receipts),
        "incomplete_transfers": sum(not r["complete"] for r in engine.receipts),
        "receipts": engine.receipts,
        "meaning": "CPU communication/compute-window planning only; steps are planned, not executed; no quality estimates",
    }
