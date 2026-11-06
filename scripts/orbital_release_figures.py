import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
BG = "#081525"
PANEL = "#10263d"
WHITE = "#edf6ff"
MUTED = "#a5bad0"
CYAN = "#56dfd0"
BLUE = "#69a9ff"
ORANGE = "#ffb56c"
RED = "#fa8d9b"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="docs/media/release_metrics.json")
    parser.add_argument("--output", default="docs/media")
    args = parser.parse_args()
    metrics_path = ROOT / args.metrics
    data = json.loads(metrics_path.read_text())
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "text.color": WHITE, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": "#456079", "svg.hashsalt": "orbital-public-release-v1"})
    files = []

    def canvas(kicker, title, subtitle):
        fig = plt.figure(figsize=(16, 9), facecolor=BG)
        fig.text(.055, .93, kicker.upper(), color=CYAN, fontsize=12, weight="bold")
        fig.text(.055, .855, title, color=WHITE, fontsize=30, weight="bold")
        fig.text(.055, .80, subtitle, color=MUTED, fontsize=14)
        return fig

    def panel(fig, bounds):
        ax = fig.add_axes(bounds, facecolor=PANEL)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    def clean(ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

    def publish(fig, name):
        for extension in ("png", "svg"):
            target = output / f"{name}.{extension}"
            fig.savefig(target, dpi=100, facecolor=BG)
            files.append(target)
        plt.close(fig)

    fig = canvas("The engineering question", "Orbital Federated Fine-Tuning",
                 "Keeping distributed training productive when communication is limited and constantly changing.")
    left = panel(fig, [.055, .19, .48, .55])
    clean(left)
    left.set_aspect("equal", adjustable="box")
    left.add_patch(Circle((.5, .48), .22, facecolor="#173f66", edgecolor=BLUE, linewidth=1.8))
    left.add_patch(Ellipse((.5, .48), .16, .43, fill=False, edgecolor="#487da4", linewidth=.8))
    left.add_patch(Ellipse((.5, .48), .43, .12, fill=False, edgecolor="#487da4", linewidth=.8))
    left.add_patch(Ellipse((.5, .48), .88, .58, angle=24, fill=False, edgecolor="#6584a1", linewidth=1, linestyle="--"))
    ground = (.52, .27)
    left.scatter(*ground, marker="^", s=140, color=CYAN, zorder=6)
    left.text(.52, .17, "Shared ground station", ha="center", fontsize=10)
    points = [(.22,.20),(.31,.77),(.70,.76),(.90,.48),(.75,.23)]
    for index, (x,y) in enumerate(points):
        color = RED if index == 4 else CYAN
        if index in (0,4):
            left.plot([x,ground[0]],[y,ground[1]], color=color, linestyle="--", alpha=.8, linewidth=1.2)
        left.add_patch(Rectangle((x-.018,y-.018),.036,.036,facecolor=WHITE, zorder=7))
        left.add_patch(Rectangle((x-.055,y-.012),.028,.024,facecolor=BLUE, zorder=7))
        left.add_patch(Rectangle((x+.027,y-.012),.028,.024,facecolor=BLUE, zorder=7))
        left.text(x,y+.045,f"Client {index}",ha="center",fontsize=9)
    left.text(.82,.13,"Link unavailable",color=RED,ha="center",fontsize=9)
    left.text(.49,.89,"Other clients wait for visibility",color=MUTED,ha="center",fontsize=10)
    right = panel(fig, [.56, .19, .385, .55])
    clean(right)
    right.text(.07,.87,"TRAIN LOCALLY. DELIVER IN TIME.",color=CYAN,fontsize=13,weight="bold")
    steps = [("01", "Five A100 clients", "A full base replica and a separate shard per client."),
             ("02", "Small adapter updates", "Frozen base; compatible client updates are combined."),
             ("03", "Communication-aware execution", "Contacts, bandwidth, outages and fixed deadlines."),
             ("04", "Verify the outcome", "Quality, useful work, recovery and measured transfers.")]
    for (number, heading, detail), y in zip(steps,[.69,.49,.29,.09]):
        right.text(.07,y,number,color=BLUE,fontsize=18,weight="bold")
        right.text(.18,y+.025,heading,fontsize=14,weight="bold")
        right.text(.18,y-.035,detail,fontsize=9.3,color=MUTED)
    fig.text(.055,.10,"REAL: GPU training and local TCP transfers     |     SIMULATED: orbital visibility, radio service and outages",fontsize=12,color=WHITE)
    fig.text(.055,.055,"Architecture schematic. GPUs ran on Earth. No flight deployment or onboard hardware feasibility is claimed.",fontsize=10,color=MUTED)
    publish(fig,"01_orbital_system")

    outage = {r["label"]:r for r in data["outage"]["results"]}
    fixed, feasible = outage["fixed"], outage["feasible_fair"]
    accepted_gain = 100 * (feasible["accepted_steps"] / fixed["accepted_steps"] - 1)
    traffic_reduction = 100 * (1 - feasible["wire_bytes"] / fixed["wire_bytes"])
    fig = canvas("A bounded outage experiment", "More useful learning reached the coordinator",
                 "One selected scenario, one training seed. Same initialization, source and maximum work budgets.")
    a = panel(fig,[.075,.25,.38,.46])
    b = panel(fig,[.565,.25,.36,.46])
    labels=["Fixed", "Feasibility / fairness"]
    a.bar(labels,[fixed["accepted_steps"],feasible["accepted_steps"]],color=[BLUE,CYAN],width=.55)
    a.set_ylim(0,18)
    a.set_ylabel("Accepted optimizer steps")
    a.set_yticks([0,4,8,12,16])
    a.grid(axis="y",alpha=.12,color=WHITE)
    a.set_axisbelow(True)
    for i,value in enumerate([fixed["accepted_steps"],feasible["accepted_steps"]]):
        a.text(i,value+.5,str(value),ha="center",fontsize=23,weight="bold")
    traffic = [fixed["wire_bytes"]/2**20,feasible["wire_bytes"]/2**20]
    b.bar(labels,traffic,color=[BLUE,CYAN],width=.55)
    b.set_ylim(0,185)
    b.set_ylabel("Simulated wire traffic (MiB)")
    b.set_yticks([0,40,80,120,160])
    b.grid(axis="y",alpha=.12,color=WHITE)
    b.set_axisbelow(True)
    for i,value in enumerate(traffic):
        b.text(i,value+5,f"{value:.3f}",ha="center",fontsize=20,weight="bold")
    fig.text(.075,.18,f"+{accepted_gain:.0f}% accepted work",color=CYAN,fontsize=22,weight="bold")
    fig.text(.565,.18,f"{traffic_reduction:.1f}% less modeled traffic",color=CYAN,fontsize=22,weight="bold")
    fig.text(.055,.105,f"Full-validation loss: fixed {fixed['validation_loss']:.4f}; feasibility/fairness {feasible['validation_loss']:.4f}; no-outage control {outage['no_outage_control']['validation_loss']:.4f}.",fontsize=12)
    fig.text(.055,.055,"These are scenario-specific results, not a general accuracy gain, throughput speedup or flight-radio measurement.",color=MUTED,fontsize=10)
    publish(fig,"02_outage_results")

    quality=data["quality"]
    fig = canvas("Quality evidence", "Better answer loss did not mean better instruction following",
                 "Identical final evaluation settings for the base, selected orbital adapter and connected federated control.")
    a=panel(fig,[.075,.25,.38,.46])
    b=panel(fig,[.565,.25,.36,.46])
    labels=["Base","Orbital","Connected"]
    losses=[r["test"]["answer_loss"] for r in quality]
    strict=[r["ifeval"]["strict"]["prompt_accuracy"]*100 for r in quality]
    a.bar(labels,losses,color=[ORANGE,CYAN,BLUE],width=.58)
    a.set_ylim(0,1.9)
    a.set_ylabel("Held-out answer loss (lower is better)")
    a.grid(axis="y",alpha=.12,color=WHITE)
    a.set_axisbelow(True)
    for i,value in enumerate(losses):
        a.text(i,value+.06,f"{value:.4f}",ha="center",fontsize=19,weight="bold")
    b.bar(labels,strict,color=[ORANGE,CYAN,BLUE],width=.58)
    b.set_ylim(0,100)
    b.set_ylabel("Strict prompt compliance (%)")
    b.set_yticks([0,20,40,60,80,100])
    b.grid(axis="y",alpha=.12,color=WHITE)
    b.set_axisbelow(True)
    for i,value in enumerate(strict):
        b.text(i,value+3,f"{value:.0f}%",ha="center",fontsize=19,weight="bold")
    fig.text(.075,.18,"365 held-out conversations",fontsize=18,weight="bold")
    fig.text(.565,.18,"100-prompt IFEval sample",fontsize=18,weight="bold")
    fig.text(.055,.105,"The connected control reached almost the same test loss as orbital training under generous communication deadlines.",fontsize=12)
    fig.text(.055,.055,"The IFEval sample does not establish a broad capability gain. It is not the complete 541-prompt benchmark.",color=MUTED,fontsize=10)
    publish(fig,"03_quality_tradeoffs")

    recovery=data["recovery"]
    payload=data["payload"]["payload_validation"]
    fig=canvas("Reliability evidence", "Correct recovery. Verified payload progress.",
               "Two separate experiments: bounded five-GPU process recovery and a CPU shared-station transfer control.")
    a=panel(fig,[.055,.22,.43,.51])
    clean(a)
    a.text(.07,.85,"FIVE-GPU PROCESS RECOVERY",color=CYAN,fontsize=13,weight="bold")
    a.text(.07,.65,"2 deliberate interruptions",fontsize=23,weight="bold")
    a.text(.07,.48,f"{recovery['max_adapter_absolute_difference']:.1f}",fontsize=48,color=CYAN,weight="bold")
    a.text(.27,.49,"maximum final adapter difference",fontsize=13)
    a.text(.07,.30,f"Validation-loss difference: {recovery['validation_loss_delta']:.1f}",fontsize=16)
    a.text(.07,.19,"Saved chunks reused; no duplicate round commits.",fontsize=12,color=MUTED)
    a.text(.07,.075,"Two rounds; one local optimizer step per client per round.",fontsize=10,color=MUTED)
    b=panel(fig,[.55,.30,.39,.37])
    rows=payload["clients"]
    b.bar([str(r["client"]) for r in rows],[r["measured_verified_payload_bytes"]/2**20 for r in rows],color=BLUE,width=.6)
    b.set_ylim(0,1.3)
    b.set_ylabel("Verified partial payload (MiB)")
    b.set_xlabel("Client")
    b.grid(axis="y",alpha=.12,color=WHITE)
    b.set_axisbelow(True)
    for i,row in enumerate(rows):
        b.text(i,row["measured_verified_payload_bytes"]/2**20+.04,f"{row['measured_verified_payload_bytes']/2**20:.1f}",ha="center",fontsize=15,weight="bold")
    fig.text(.55,.70,"SHARED-STATION PAYLOAD CONTROL",color=CYAN,fontsize=13,weight="bold")
    fig.text(.55,.21,"5 clients · 9 chunks · 4.5 MiB verified",fontsize=17,weight="bold")
    fig.text(.55,.16,"Exact predicted progress; exclusive station service.",fontsize=12,color=MUTED)
    fig.text(.055,.095,"Recovery covers two injected process interruptions. The payload control verifies partial chunks, not complete new model updates.",fontsize=11)
    fig.text(.055,.05,"No hardware power-loss, radiation qualification, flight deployment or general onboard-training feasibility is claimed.",color=MUTED,fontsize=10)
    publish(fig,"04_recovery_and_transfers")
    manifest={"metrics_sha256":hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
              "files":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"images":4,"formats":["png","svg"],"output":str(output)},indent=2))


if __name__ == "__main__":
    main()
