import copy

from .feasibility import POLICIES


def plan_round(engine, clients, start, deadline, payload_bytes, step_seconds, max_steps, policy, age, round_id):
    if policy not in POLICIES or min(payload_bytes, step_seconds, max_steps) <= 0 or deadline <= start:
        raise ValueError("Invalid planning inputs")
    tasks = []
    order = list(range(clients))
    if policy == "contact_aware":
        order.sort(key=lambda i: (min((c["end"] for c in engine.contacts if c["client"] == i and c["end"] > start), default=float("inf")), i))
    if policy == "feasible_fair":
        pending = set(order)
        while pending:
            proposals = []
            for client in sorted(pending):
                for steps in sorted({1, max(1, max_steps // 2), max_steps}, reverse=True):
                    trial = copy.deepcopy(engine)
                    down_id, up_id = f"r{round_id}-c{client}-down", f"r{round_id}-c{client}-up"
                    down = trial.transfer(client, payload_bytes, start, deadline, "download", down_id)
                    if not down["complete"]:
                        continue
                    complete = down["arrival"] + steps * step_seconds
                    if complete >= deadline:
                        continue
                    up = trial.transfer(client, payload_bytes, complete, deadline, "upload", up_id)
                    if up["complete"]:
                        proposals.append((-age[client], up["arrival"], client, steps, trial, down_id, up_id, down["arrival"], complete))
                        break
            if not proposals:
                break
            _, _, client, steps, engine, down_id, up_id, ready, complete = min(proposals, key=lambda p: p[:3])
            pending.remove(client)
            tasks.append({"client": client, "steps": steps, "download_id": down_id, "upload_id": up_id, "ready": ready, "complete": complete})
    else:
        for client in order:
            down_id = f"r{round_id}-c{client}-down"
            down = engine.transfer(client, payload_bytes, start, deadline, "download", down_id)
            if not down["complete"]:
                continue
            complete = down["arrival"] + max_steps * step_seconds
            if complete <= deadline:
                tasks.append({"client": client, "steps": max_steps, "download_id": down_id, "upload_id": f"r{round_id}-c{client}-up", "ready": down["arrival"], "complete": complete})
        for task in sorted(tasks, key=lambda t: (t["complete"], t["client"])):
            engine.transfer(task["client"], payload_bytes, task["complete"], deadline, "upload", task["upload_id"])
    receipts = {r["id"]: r for r in engine.receipts}
    uploads = [task for task in tasks if receipts[task["upload_id"]]["complete"]]
    succeeded = len(uploads) >= engine.settings.min_participants
    accepted = {task["client"] for task in uploads} if succeeded else set()
    summary = {
        "round": round_id, "start": start, "deadline": deadline,
        "delivered_clients": [t["client"] for t in uploads], "accepted_clients": sorted(accepted),
        "scheduled_steps": sum(t["steps"] for t in tasks),
        "accepted_steps": sum(t["steps"] for t in uploads) if succeeded else 0,
        "successful": succeeded,
        "last_upload_arrival": max((receipts[t["upload_id"]]["arrival"] for t in uploads), default=None),
    }
    return engine, tasks, summary
