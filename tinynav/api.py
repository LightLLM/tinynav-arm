from __future__ import annotations

import platform

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .controller import SafetyController
from .simulator import SCENARIOS, WarehouseSimulator

app = FastAPI(title="TinyNav-Arm API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://localhost:5173"], allow_methods=["GET"], allow_headers=["*"])
simulator = WarehouseSimulator()
controller = SafetyController()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "architecture": platform.machine(), "mode": "simulation"}


@app.get("/api/frame/{scenario}")
def frame(scenario: str) -> dict:
    if scenario not in SCENARIOS:
        raise HTTPException(404, f"Unknown scenario. Choose: {', '.join(SCENARIOS)}")
    sample = simulator.next_frame(scenario)
    decision = controller.decide(sample.detections)
    return {
        "timestamp": sample.timestamp,
        "scenario": scenario,
        "detections": [d.to_dict() for d in sample.detections],
        "decision": decision.to_dict(),
        "source": "deterministic-simulation",
    }
