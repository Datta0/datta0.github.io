"""Generate editable dark SVG diagrams and a matching Excalidraw source file.

The blog references the SVG files directly, while the Excalidraw file is kept as
the editable source of record for future changes. The SVGs intentionally avoid
embedded rasters so they remain inspectable and editable as vector graphics.
"""

from __future__ import annotations

import html
import json
import os
import random
from dataclasses import dataclass, field
from typing import Iterable


random.seed(20260530)

BASE = os.path.dirname(__file__)

BG = "#05070b"
PANEL = "#0e141f"
PANEL_2 = "#111827"
TEXT = "#eef2ff"
MUTED = "#9aa7bd"
GRID = "#1f2937"
GREEN = "#47d16c"
BLUE = "#60a5fa"
CYAN = "#22d3ee"
PINK = "#f472b6"
ORANGE = "#f59e0b"
YELLOW = "#facc15"
LIME = "#a3e635"
RED = "#fb7185"
PURPLE = "#c084fc"
BROWN = "#b08968"
WHITE = "#f8fafc"


def rid(prefix: str = "e") -> str:
    return prefix + "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=15))


def esc(value: str) -> str:
    return html.escape(value, quote=True)


@dataclass
class SvgDoc:
    width: int
    height: int
    title: str
    parts: list[str] = field(default_factory=list)
    ops: list[dict] = field(default_factory=list)

    def add(self, raw: str) -> None:
        self.parts.append(raw)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = PANEL,
        stroke: str = GRID,
        sw: float = 1.5,
        rx: float = 8,
        dash: str | None = None,
        opacity: float = 1,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.ops.append({
            "kind": "rect", "x": x, "y": y, "w": w, "h": h, "fill": fill,
            "stroke": stroke, "sw": sw, "dash": dash,
        })
        self.add(
            f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx:g}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}"{dash_attr} opacity="{opacity:g}"/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = MUTED,
        sw: float = 2,
        dash: str | None = None,
        marker: bool = False,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker_attr = ' marker-end="url(#arrow)"' if marker else ""
        self.ops.append({
            "kind": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "stroke": stroke, "sw": sw, "dash": dash, "marker": marker,
        })
        self.add(
            f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
            f'stroke="{stroke}" stroke-width="{sw:g}" stroke-linecap="round"{dash_attr}{marker_attr}/>'
        )

    def path(
        self,
        d: str,
        stroke: str = MUTED,
        sw: float = 2,
        fill: str = "none",
        dash: str | None = None,
        marker: bool = False,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker_attr = ' marker-end="url(#arrow)"' if marker else ""
        self.ops.append({
            "kind": "path", "d": d, "stroke": stroke, "sw": sw,
            "fill": fill, "dash": dash, "marker": marker,
        })
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash_attr}{marker_attr}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: int = 18,
        color: str = TEXT,
        weight: int = 500,
        anchor: str = "start",
    ) -> None:
        self.ops.append({
            "kind": "text", "x": x, "y": y, "value": value, "size": size,
            "color": color, "anchor": anchor,
        })
        self.add(
            f'<text x="{x:g}" y="{y:g}" fill="{color}" font-size="{size:g}" '
            f'font-family="Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif" '
            f'font-weight="{weight:g}" text-anchor="{anchor}">{esc(value)}</text>'
        )

    def small(self, x: float, y: float, value: str, color: str = MUTED, anchor: str = "start") -> None:
        self.text(x, y, value, 14, color, 450, anchor)

    def title_block(self, title: str, subtitle: str) -> None:
        self.text(34, 48, title, 26, TEXT, 750)
        self.small(34, 74, subtitle)

    def tostring(self) -> str:
        body = "\n  ".join(self.parts)
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="title desc">
  <title id="title">{esc(self.title)}</title>
  <desc id="desc">Dark vector diagram for the Systems for LLM RL blog.</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/>
    </marker>
    <linearGradient id="memfade" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0" stop-color="#172033"/>
      <stop offset="1" stop-color="#0b1020"/>
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="{BG}"/>
  {body}
</svg>
'''


def pill(doc: SvgDoc, x: float, y: float, w: float, label: str, color: str, sub: str | None = None) -> None:
    doc.rect(x, y, w, 54 if sub else 42, "#101827", color, 1.8, 7)
    doc.text(x + w / 2, y + 26, label, 16, TEXT, 700, "middle")
    if sub:
        doc.small(x + w / 2, y + 45, sub, MUTED, "middle")


def box(doc: SvgDoc, x: float, y: float, w: float, h: float, label: str, color: str, sub: str | None = None) -> None:
    doc.rect(x, y, w, h, "#0f172a", color, 2, 7)
    doc.text(x + 16, y + 28, label, 17, TEXT, 700)
    if sub:
        for i, line in enumerate(sub.split("\n")):
            doc.small(x + 16, y + 52 + i * 19, line)


def mem_stack(doc: SvgDoc, x: float, y: float, title: str, items: Iterable[tuple[str, float, str]]) -> None:
    scale = 2.3
    width = 188
    pad = 9
    total_h = sum(gb * scale for _, gb, _ in items) + pad * 2
    doc.text(x, y - 14, title, 17, GREEN, 700)
    doc.rect(x - 12, y, width + 24, total_h, "url(#memfade)", GREEN, 2.2, 8)
    cur = y + pad
    for label, gb, color in items:
        h = max(17, gb * scale)
        doc.rect(x, cur, width, h, color, "#000000", 0.4, 5, opacity=0.92)
        doc.text(x + 10, cur + min(h - 5, h / 2 + 5), f"{label}  {gb:g} GB", 13, "#06111f", 750)
        cur += h


def phase_band(doc: SvgDoc, x: float, y: float, w: float, label: str, color: str) -> None:
    doc.rect(x, y, w, 72, "#0f172a", color, 2, 7)
    doc.text(x + w / 2, y + 30, label, 17, TEXT, 750, "middle")
    doc.small(x + w / 2, y + 53, "active HBM owner", MUTED, "middle")


def write_svg(name: str, doc: SvgDoc) -> None:
    with open(os.path.join(BASE, name), "w", encoding="utf-8") as f:
        f.write(doc.tostring())


def diagram_naive() -> SvgDoc:
    d = SvgDoc(960, 600, "Naive colocated rollout and training memory")
    d.title_block("Naive colocation: memory adds up", "Rollout and trainer processes both stay resident on the same GPU pool.")
    mem_stack(d, 92, 136, "Inference engine", [
        ("vLLM weights", 16, BROWN), ("CUDA graphs", 2, ORANGE), ("KV cache", 24, BLUE), ("activations", 3, PURPLE)
    ])
    d.text(480, 300, "+", 58, MUTED, 750, "middle")
    mem_stack(d, 636, 136, "Trainer", [
        ("weights", 16, BROWN), ("master weights", 32, WHITE), ("gradients", 16, CYAN), ("optimizer states", 64, PINK), ("activations", 5, PURPLE)
    ])
    doc_note(d, 278, 510, 404, "Duplicate base weights plus all runtime state compete for HBM.")
    return d


def diagram_step() -> SvgDoc:
    d = SvgDoc(960, 520, "Online RL step split between inference and training systems")
    d.title_block("Online RL lifecycle", "The systems alternate between rollout generation and policy updates.")
    xs = [70, 250, 430, 610, 790]
    labels = [
        ("sync weights", BLUE, "trainer -> vLLM"),
        ("rollouts", GREEN, "sample completions"),
        ("forward loss", PURPLE, "logprobs, rewards"),
        ("backward", CYAN, "gradients"),
        ("optimizer step", PINK, "new policy"),
    ]
    for i, (x, (label, color, sub)) in enumerate(zip(xs, labels)):
        box(d, x, 170, 125, 92, label, color, sub)
        if i < len(xs) - 1:
            d.line(x + 125, 216, xs[i + 1] - 10, 216, MUTED, 2.2, marker=True)
    d.path("M 855 158 C 900 92 108 92 86 158", BLUE, 2.2, dash="7 7", marker=True)
    d.small(480, 106, "repeat with fresher weights", BLUE, "middle")
    doc_note(d, 246, 374, 468, "Inference and training want the same weights, but not at the same time.")
    return d


def diagram_sleep_timeslices() -> SvgDoc:
    d = SvgDoc(960, 600, "Sleep mode timeslices GPU memory")
    d.title_block("Sleep mode: temporal memory sharing", "Only the active phase keeps its large working set in HBM.")
    d.line(100, 160, 860, 160, GRID, 2)
    phase_band(d, 100, 190, 300, "rollout: vLLM awake", GREEN)
    phase_band(d, 560, 190, 300, "train: vLLM asleep", CYAN)
    d.line(400, 226, 560, 226, MUTED, 2.2, marker=True)
    d.small(480, 212, "sleep / wake", BLUE, "middle")
    mem_stack(d, 138, 312, "HBM during rollout", [
        ("vLLM weights", 16, BROWN), ("KV cache", 24, BLUE), ("CUDA graphs", 2, ORANGE), ("activations", 3, PURPLE)
    ])
    mem_stack(d, 598, 312, "HBM during train", [
        ("weights", 16, BROWN), ("master", 32, WHITE), ("gradients", 16, CYAN), ("optimizer", 64, PINK), ("activations", 5, PURPLE)
    ])
    doc_note(d, 318, 520, 324, "Target envelope: max(inference, trainer), not inference + trainer.")
    return d


def diagram_startup_sleep() -> SvgDoc:
    d = SvgDoc(960, 560, "vLLM cold restart compared with sleep and wake")
    d.title_block("Cold restart vs sleep mode", "Avoid paying startup cost on every RL step.")
    bars = [
        ("load weights", 75, BROWN),
        ("profile", 32, BLUE),
        ("CUDA graph capture", 9, ORANGE),
        ("sleep level 2", 0.24, GREEN),
        ("wake", 0.4, CYAN),
    ]
    max_v = 80
    x0 = 120
    for i, (label, value, color) in enumerate(bars):
        y = 145 + i * 66
        w = max(7, value / max_v * 650)
        d.small(40, y + 24, label)
        d.rect(x0, y, w, 36, color, color, 0, 5)
        d.text(x0 + w + 14, y + 25, f"{value:g}s", 16, TEXT, 700)
    d.line(x0, 484, x0 + 650, 484, GRID, 2)
    d.small(x0 + 650, 508, "seconds", MUTED, "middle")
    doc_note(d, 510, 408, 330, "Sleep keeps allocations warm enough to dodge the multi-minute restart path.")
    return d


def diagram_chunked_loss() -> SvgDoc:
    d = SvgDoc(960, 560, "Chunked GRPO loss streams token losses")
    d.title_block("Chunked loss streaming", "Keep hidden states; stream logits through the loss instead of materializing them.")
    box(d, 66, 176, 178, 100, "hidden states", PURPLE, "batch x seq x hidden\nmanageable")
    box(d, 392, 154, 190, 144, "full logits", RED, "batch x seq x vocab\nlarge temporary")
    d.line(244, 226, 392, 226, RED, 2.2, dash="8 7", marker=True)
    d.text(317, 206, "avoid", 15, RED, 750, "middle")
    for i in range(6):
        x = 322 + i * 44
        d.rect(x, 382, 30, 58, "#132033", BLUE, 1.5, 5)
        d.text(x + 15, 417, str(i + 1), 14, TEXT, 750, "middle")
    d.line(244, 254, 322, 410, BLUE, 2.2, marker=True)
    d.line(586, 410, 740, 410, BLUE, 2.2, marker=True)
    box(d, 740, 360, 154, 100, "accumulate", GREEN, "token losses\nsmall scalar state")
    doc_note(d, 272, 496, 416, "GRPO decomposes over tokens, so logits can be produced and consumed in chunks.")
    return d


def diagram_lora_share() -> SvgDoc:
    d = SvgDoc(960, 560, "LoRA weight sharing uses one frozen base model")
    d.title_block("LoRA weight sharing", "The frozen base model can be shared; only the adapter trains.")
    pill(d, 80, 220, 150, "vLLM", BLUE, "rollout")
    pill(d, 80, 316, 150, "trainer", CYAN, "loss + grads")
    box(d, 390, 228, 220, 116, "shared base weights", BROWN, "one resident frozen copy\npointers reused")
    pill(d, 694, 218, 150, "LoRA adapter", LIME, "trainable")
    pill(d, 694, 306, 150, "tiny opt state", PINK, "adapter only")
    d.line(230, 247, 390, 264, MUTED, 2.2, marker=True)
    d.line(230, 343, 390, 312, MUTED, 2.2, marker=True)
    d.line(610, 286, 694, 244, MUTED, 2.2, marker=True)
    d.line(610, 306, 694, 332, MUTED, 2.2, marker=True)
    doc_note(d, 263, 458, 434, "No second full-weight copy is needed when the base is frozen.")
    return d


def diagram_weight_sleep() -> SvgDoc:
    d = SvgDoc(960, 600, "Weight sharing with vLLM sleep mode")
    d.title_block("Weight sharing + sleep mode", "Sleep frees transient inference state while the shared base stays resident.")
    mem_stack(d, 110, 160, "Rollout", [
        ("shared base", 16, BROWN), ("KV cache", 24, BLUE), ("CUDA graphs", 2, ORANGE), ("LoRA adapter", 1, LIME)
    ])
    d.line(360, 270, 510, 270, MUTED, 2.2, marker=True)
    d.small(435, 252, "sleep", BLUE, "middle")
    mem_stack(d, 570, 160, "Training", [
        ("shared base", 16, BROWN), ("LoRA grads", 1, CYAN), ("LoRA opt", 1, PINK), ("activations", 3, PURPLE)
    ])
    d.rect(570, 338, 188, 60, "#101827", GRID, 1.5, 7, dash="7 6")
    d.text(664, 374, "KV cache freed", 14, MUTED, 650, "middle")
    doc_note(d, 272, 510, 416, "Use sleep behavior that does not discard the shared base weights.")
    return d


def diagram_async() -> SvgDoc:
    d = SvgDoc(960, 600, "Async GRPO pipeline overlaps rollout and training")
    d.title_block("Async GRPO pipeline", "Disaggregated GPUs trade small policy lag for higher utilization.")
    lanes = [("inference GPU", 160, GREEN), ("trainer GPU", 300, CYAN), ("weight sync", 440, BLUE)]
    for label, y, color in lanes:
        d.small(62, y + 26, label, color)
        d.line(190, y + 20, 870, y + 20, GRID, 2)
    blocks = [
        (210, 152, 170, 56, "rollout batch N", GREEN),
        (430, 152, 170, 56, "rollout N+1", GREEN),
        (650, 152, 170, 56, "rollout N+2", GREEN),
        (360, 292, 170, 56, "train batch N", CYAN),
        (580, 292, 170, 56, "train N+1", CYAN),
        (250, 432, 120, 50, "sync w0", BLUE),
        (505, 432, 120, 50, "sync w1", BLUE),
        (735, 432, 120, 50, "sync w2", BLUE),
    ]
    for x, y, w, h, label, color in blocks:
        d.rect(x, y, w, h, "#0f172a", color, 2, 7)
        d.text(x + w / 2, y + 34, label, 15, TEXT, 700, "middle")
    d.path("M 380 180 C 410 215 370 250 445 292", MUTED, 2, marker=True)
    d.path("M 600 180 C 630 215 590 250 665 292", MUTED, 2, marker=True)
    d.small(758, 270, "policy lag: 1-2 steps", ORANGE, "middle")
    doc_note(d, 260, 530, 440, "Corrections and clipping keep small staleness from becoming fully off-policy training.")
    return d


def diagram_tradeoff() -> SvgDoc:
    d = SvgDoc(960, 640, "Tradeoff map for LLM RL systems optimizations")
    d.title_block("Systems tradeoff map", "Each optimization moves along VRAM, overhead, and policy-freshness axes.")
    origin = (145, 520)
    d.line(origin[0], origin[1], 850, origin[1], MUTED, 2.4, marker=True)
    d.line(origin[0], origin[1], origin[0], 126, MUTED, 2.4, marker=True)
    d.small(842, 552, "more throughput / utilization", MUTED, "end")
    d.small(122, 130, "less VRAM pressure", MUTED, "middle")
    points = [
        (220, 455, "naive colocate", RED),
        (330, 300, "sleep mode", GREEN),
        (455, 242, "LoRA sharing", LIME),
        (585, 350, "chunked loss", PURPLE),
        (720, 220, "async GRPO", BLUE),
        (690, 450, "disaggregated", CYAN),
    ]
    for x, y, label, color in points:
        d.rect(x - 8, y - 8, 16, 16, color, color, 0, 4)
        d.text(x + 16, y + 5, label, 15, TEXT, 700)
    d.path("M 220 455 C 320 360 350 320 455 242 C 560 210 650 210 720 220", GREEN, 2.2, dash="8 7", marker=True)
    d.small(564, 128, "freshness cost rises as work overlaps or decouples", ORANGE, "middle")
    doc_note(d, 284, 574, 392, "The best point depends on model size, sequence length, topology, and acceptable staleness.")
    return d


def doc_note(doc: SvgDoc, x: float, y: float, w: float, value: str) -> None:
    doc.rect(x, y - 28, w, 48, "#0b1120", "#263244", 1.2, 7)
    doc.small(x + w / 2, y + 2, value, MUTED, "middle")


DIAGRAMS = [
    ("naive_colocation_memory.svg", diagram_naive),
    ("rl_step_two_systems.svg", diagram_step),
    ("sleep_mode_timeslices.svg", diagram_sleep_timeslices),
    ("startup_vs_sleep.svg", diagram_startup_sleep),
    ("chunked_loss_streaming.svg", diagram_chunked_loss),
    ("lora_weight_sharing.svg", diagram_lora_share),
    ("weight_sharing_sleep_mode.svg", diagram_weight_sleep),
    ("async_grpo_pipeline.svg", diagram_async),
    ("rl_systems_tradeoff_map.svg", diagram_tradeoff),
]


def excalidraw_rect(x: float, y: float, w: float, h: float, stroke: str, fill: str, frame_id: str | None) -> dict:
    return {
        "id": rid("r"),
        "type": "rectangle",
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": stroke,
        "backgroundColor": fill,
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": frame_id,
        "roundness": {"type": 3},
        "seed": random.randint(1, 2**31),
        "version": 1,
        "versionNonce": random.randint(1, 2**31),
        "isDeleted": False,
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
    }


def excalidraw_text(x: float, y: float, value: str, size: int, color: str, frame_id: str | None) -> dict:
    return {
        "id": rid("t"),
        "type": "text",
        "x": x,
        "y": y,
        "width": max(80, int(len(value) * size * 0.58)),
        "height": int(size * 1.25),
        "angle": 0,
        "strokeColor": color,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": frame_id,
        "roundness": None,
        "seed": random.randint(1, 2**31),
        "version": 1,
        "versionNonce": random.randint(1, 2**31),
        "isDeleted": False,
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
        "fontSize": size,
        "fontFamily": 1,
        "text": value,
        "textAlign": "left",
        "verticalAlign": "top",
        "baseline": int(size * 0.9),
        "containerId": None,
        "originalText": value,
        "lineHeight": 1.25,
    }


def excalidraw_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str,
    frame_id: str | None,
    marker: bool = False,
    dash: str | None = None,
) -> dict:
    return {
        "id": rid("a" if marker else "l"),
        "type": "arrow" if marker else "line",
        "x": x1,
        "y": y1,
        "width": x2 - x1,
        "height": y2 - y1,
        "angle": 0,
        "strokeColor": stroke,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "dashed" if dash else "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": frame_id,
        "roundness": {"type": 2},
        "seed": random.randint(1, 2**31),
        "version": 1,
        "versionNonce": random.randint(1, 2**31),
        "isDeleted": False,
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
        "points": [[0, 0], [x2 - x1, y2 - y1]],
        "lastCommittedPoint": None,
        "startBinding": None,
        "endBinding": None,
        "startArrowhead": None,
        "endArrowhead": "arrow" if marker else None,
    }


def add_doc_to_excalidraw(elements: list[dict], doc: SvgDoc, ox: float, oy: float, frame_id: str) -> None:
    for op in doc.ops:
        kind = op["kind"]
        if kind == "rect":
            fill = op["fill"] if not str(op["fill"]).startswith("url(") else PANEL_2
            el = excalidraw_rect(
                ox + op["x"], oy + op["y"], op["w"], op["h"],
                op["stroke"], fill, frame_id
            )
            if op.get("dash"):
                el["strokeStyle"] = "dashed"
            elements.append(el)
        elif kind == "line":
            elements.append(excalidraw_line(
                ox + op["x1"], oy + op["y1"], ox + op["x2"], oy + op["y2"],
                op["stroke"], frame_id, bool(op.get("marker")), op.get("dash")
            ))
        elif kind == "text":
            x = ox + op["x"]
            if op.get("anchor") == "middle":
                x -= max(40, len(op["value"]) * op["size"] * 0.29)
            elif op.get("anchor") == "end":
                x -= max(80, len(op["value"]) * op["size"] * 0.58)
            elements.append(excalidraw_text(x, oy + op["y"] - op["size"], op["value"], op["size"], op["color"], frame_id))


def build_excalidraw() -> dict:
    elements: list[dict] = []
    frame_w, frame_h = 960, 640
    gap = 80
    for index, (filename, builder) in enumerate(DIAGRAMS):
        row = index // 3
        col = index % 3
        x = col * (frame_w + gap)
        y = row * (frame_h + gap)
        fid = rid("f")
        title = filename.replace(".svg", "").replace("_", " ")
        frame = excalidraw_rect(x, y, frame_w, frame_h, "#334155", BG, None)
        frame.update({"id": fid, "type": "frame", "name": title, "roundness": None})
        elements.append(frame)
        add_doc_to_excalidraw(elements, builder(), x, y, fid)
    return {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {"gridSize": 20, "viewBackgroundColor": BG},
        "files": {},
    }


def main() -> None:
    for filename, builder in DIAGRAMS:
        write_svg(filename, builder())
    with open(os.path.join(BASE, "rl_systems_diagrams.excalidraw"), "w", encoding="utf-8") as f:
        json.dump(build_excalidraw(), f, indent=2)
        f.write("\n")
    print(f"Wrote {len(DIAGRAMS)} SVGs and rl_systems_diagrams.excalidraw")


if __name__ == "__main__":
    main()
