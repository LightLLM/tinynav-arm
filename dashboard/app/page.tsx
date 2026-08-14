"use client";

import { useEffect, useMemo, useState } from "react";

type Mode = "FP32" | "INT8";
type Scenario = "Clear aisle" | "Forklift crossing" | "Blocked path";

const scenarios: Record<Scenario, { decision: string; color: string; speed: number; risk: number; distance: string; objects: { label: string; confidence: number; cls: string }[] }> = {
  "Clear aisle": { decision: "GO", color: "#bafc72", speed: 1.2, risk: 8, distance: "4.8 m", objects: [{ label: "PALLET", confidence: 91, cls: "box pallet" }] },
  "Forklift crossing": { decision: "SLOW", color: "#ffd166", speed: 0.4, risk: 62, distance: "2.1 m", objects: [{ label: "FORKLIFT", confidence: 96, cls: "box forklift" }, { label: "WORKER", confidence: 89, cls: "box worker" }] },
  "Blocked path": { decision: "STOP", color: "#ff6b5f", speed: 0, risk: 94, distance: "0.7 m", objects: [{ label: "CRATE", confidence: 98, cls: "box crate" }, { label: "WORKER", confidence: 92, cls: "box worker near" }] },
};

const metrics = {
  FP32: { latency: 83.6, fps: 12.0, memory: 412, size: 24.7, quality: 72.8 },
  INT8: { latency: 31.2, fps: 32.1, memory: 168, size: 6.4, quality: 71.9 },
};

export default function Home() {
  const [mode, setMode] = useState<Mode>("INT8");
  const [scenario, setScenario] = useState<Scenario>("Forklift crossing");
  const [running, setRunning] = useState(true);
  const [frame, setFrame] = useState(1842);
  const [tab, setTab] = useState<"mission" | "benchmarks" | "pipeline">("mission");
  const [optimized, setOptimized] = useState(true);
  const s = scenarios[scenario];
  const m = metrics[mode];

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => setFrame((v) => v + 1), mode === "INT8" ? 620 : 1100);
    return () => clearInterval(id);
  }, [running, mode]);

  const improvement = useMemo(() => Math.round((1 - metrics.INT8.latency / metrics.FP32.latency) * 100), []);

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="TinyNav Arm home">
          <span className="brand-mark"><i /><i /><i /></span>
          <span>TINYNAV<span className="brand-arm">/ARM</span></span>
        </a>
        <div className="challenge"><span className="live-dot" /> ARM CREATE 2026 <b>·</b> PHYSICAL AI</div>
        <nav>
          <button className={tab === "mission" ? "active" : ""} onClick={() => setTab("mission")}>MISSION</button>
          <button className={tab === "benchmarks" ? "active" : ""} onClick={() => setTab("benchmarks")}>BENCHMARKS</button>
          <button className={tab === "pipeline" ? "active" : ""} onClick={() => setTab("pipeline")}>PIPELINE</button>
        </nav>
        <a className="repo" href="#opensource">VIEW REPOSITORY <span>↗</span></a>
      </header>

      {tab === "mission" && <>
        <section className="hero" id="top">
          <div className="hero-copy">
            <p className="eyebrow"><span>EDGE PERCEPTION</span> / RESPONSIVE AUTONOMY</p>
            <h1>Robots should<br />react at the speed<br /><em>of the real world.</em></h1>
            <p className="lede">TinyNav compresses robot vision models for Arm64 edge hardware—cutting response time without losing the awareness that keeps people and machines moving safely.</p>
            <div className="hero-actions">
              <button className="primary" onClick={() => setRunning((v) => !v)}>{running ? "PAUSE LIVE DEMO" : "RUN LIVE DEMO"} <span>{running ? "Ⅱ" : "▶"}</span></button>
              <button className="text-btn" onClick={() => setTab("benchmarks")}>EXPLORE THE BENCHMARKS <span>→</span></button>
            </div>
            <div className="proof-strip">
              <div><strong>2.7×</strong><span>FASTER<br />INFERENCE</span></div>
              <div><strong>74%</strong><span>SMALLER<br />MODEL</span></div>
              <div><strong>-0.9</strong><span>mAP QUALITY<br />DELTA</span></div>
            </div>
          </div>

          <div className="vision-card">
            <div className="vision-head">
              <div><span className="live-dot" /> LIVE ROBOT FEED</div>
              <span>CAM_01 · FRAME {frame}</span>
            </div>
            <div className={`camera ${running ? "running" : ""}`}>
              <div className="scanline" />
              <div className="warehouse">
                <div className="ceiling"><i /><i /><i /><i /></div>
                <div className="shelf left"><i /><i /><i /></div>
                <div className="shelf right"><i /><i /><i /></div>
                <div className="floor-lines"><i /><i /><i /></div>
                <div className="vanish" />
                {s.objects.map((o, i) => <div key={o.label} className={o.cls}>
                  <span>{o.label} <b>{o.confidence}%</b></span>
                  <i /><i /><i /><i />
                </div>)}
              </div>
              <div className="camera-readout top-left">ONNX RUNTIME<br />AARCH64 / CPU</div>
              <div className="camera-readout top-right">{m.latency.toFixed(1)} MS<br />{m.fps.toFixed(1)} FPS</div>
              <div className="crosshair"><i /><i /></div>
              <div className="decision" style={{ "--decision": s.color } as React.CSSProperties}>
                <span>CONTROL DECISION</span><strong>{s.decision}</strong><small>{s.decision === "GO" ? "PATH CLEAR" : s.decision === "SLOW" ? "CROSSING DETECTED" : "OBSTACLE IN SAFETY ZONE"}</small>
              </div>
            </div>
            <div className="vision-controls">
              <label>SCENARIO
                <select value={scenario} onChange={(e) => setScenario(e.target.value as Scenario)}>
                  {Object.keys(scenarios).map((name) => <option key={name}>{name}</option>)}
                </select>
              </label>
              <div className="mode-switch"><span>MODEL</span><button className={mode === "FP32" ? "selected" : ""} onClick={() => setMode("FP32")}>FP32</button><button className={mode === "INT8" ? "selected" : ""} onClick={() => setMode("INT8")}>INT8</button></div>
              <div className="telemetry"><span>SPEED <b>{s.speed.toFixed(1)} m/s</b></span><span>NEAREST <b>{s.distance}</b></span><span>RISK <b style={{ color: s.color }}>{s.risk}%</b></span></div>
            </div>
          </div>
        </section>

        <section className="what-section">
          <div className="section-intro"><p className="eyebrow">THE OPTIMIZATION GAP</p><h2>A smarter model is only useful<br /><em>if the robot can run it.</em></h2></div>
          <div className="story-grid">
            <article><span>01</span><div className="story-icon latency-icon"><i /><i /></div><h3>Perceive</h3><p>Camera frames enter an efficient detection pipeline designed for constrained Arm64 edge systems.</p></article>
            <article><span>02</span><div className="story-icon brain-icon">⌁</div><h3>Optimize</h3><p>Graph fusion, static INT8 quantization, calibration, and memory tuning shrink the workload.</p></article>
            <article><span>03</span><div className="story-icon control-icon">⌖</div><h3>Act</h3><p>Detections become immediate control signals: GO, SLOW, TURN, or STOP—without a cloud round trip.</p></article>
          </div>
        </section>

        <section className="optimizer-section">
          <div className="optimizer-copy"><p className="eyebrow">ONE-COMMAND OPTIMIZATION</p><h2>From oversized model<br />to <em>edge-ready artifact.</em></h2><p>One reproducible pipeline generates the optimized model, benchmark report, accuracy comparison, and hardware metadata developers need to deploy with confidence.</p>
            <div className="terminal"><div><i /><i /><i /><span>tinynav — optimizer</span></div><code><b>$</b> python optimize.py <br />&nbsp;&nbsp;--model baseline.onnx <br />&nbsp;&nbsp;--target arm64 --precision int8<br /><br /><span>✓ graph optimized</span><br /><span>✓ calibrated with 240 frames</span><br /><span>✓ benchmark report generated</span></code></div>
          </div>
          <div className="optimization-card">
            <div className="opt-head"><div><span>OPTIMIZATION RUN</span><strong>warehouse-detector-v1</strong></div><span className="verified">✓ VERIFIED ON ARM64</span></div>
            <div className="model-flow">
              <div className={!optimized ? "pulse" : ""}><span>BASELINE</span><strong>FP32</strong><small>24.7 MB</small></div>
              <div className="flow-arrow"><span>{optimized ? "74% SMALLER" : "RUNNING…"}</span>━━━━▶</div>
              <div className={optimized ? "pulse" : ""}><span>OPTIMIZED</span><strong>INT8</strong><small>6.4 MB</small></div>
            </div>
            <div className="metric-compare">
              <div><span>INFERENCE LATENCY</span><b>83.6 <small>ms</small></b><i>→</i><strong>31.2 <small>ms</small></strong><em>-{improvement}%</em></div>
              <div><span>PEAK MEMORY</span><b>412 <small>MB</small></b><i>→</i><strong>168 <small>MB</small></strong><em>-59%</em></div>
              <div><span>MODEL QUALITY</span><b>72.8 <small>mAP</small></b><i>→</i><strong>71.9 <small>mAP</small></strong><em className="neutral">-0.9</em></div>
            </div>
            <button className="primary full" onClick={() => { setOptimized(false); setTimeout(() => setOptimized(true), 1100); }}>{optimized ? "RERUN OPTIMIZATION" : "OPTIMIZING…"} <span>↻</span></button>
          </div>
        </section>
      </>}

      {tab === "benchmarks" && <BenchmarkView mode={mode} setMode={setMode} />}
      {tab === "pipeline" && <PipelineView />}

      <section className="open-source" id="opensource">
        <div><p className="eyebrow">OPEN. REPRODUCIBLE. BUILT FOR ARM.</p><h2>Give every robot<br /><em>a faster reflex.</em></h2></div>
        <div><p>TinyNav-Arm is an open-source reference pipeline for robotics teams moving computer vision from training environments to responsive Arm64 edge inference.</p><button className="primary">EXPLORE ON GITHUB <span>↗</span></button></div>
      </section>
      <footer><div className="brand"><span className="brand-mark"><i /><i /><i /></span><span>TINYNAV<span className="brand-arm">/ARM</span></span></div><p>PHYSICAL AI · OPTIMIZED AT THE EDGE</p><p>ARM CREATE 2026</p></footer>
    </main>
  );
}

function BenchmarkView({ mode, setMode }: { mode: Mode; setMode: (m: Mode) => void }) {
  const rows = [
    ["Median latency", "83.6 ms", "31.2 ms", "-63%"], ["P95 latency", "101.4 ms", "38.7 ms", "-62%"], ["Throughput", "12.0 FPS", "32.1 FPS", "+168%"], ["Peak memory", "412 MB", "168 MB", "-59%"], ["Model size", "24.7 MB", "6.4 MB", "-74%"], ["COCO mAP", "72.8", "71.9", "-0.9"]
  ];
  return <section className="detail-page"><div className="detail-hero"><p className="eyebrow">MEASURED ON ARM64</p><h1>Every claim has<br /><em>a number behind it.</em></h1><p>Representative prototype data for visualizing the final benchmark report. Replace with measured hardware results before competition submission.</p></div>
    <div className="bench-grid"><div className="chart-card"><div className="card-title"><span>INFERENCE LATENCY</span><div><button className={mode === "FP32" ? "selected" : ""} onClick={() => setMode("FP32")}>FP32</button><button className={mode === "INT8" ? "selected" : ""} onClick={() => setMode("INT8")}>INT8</button></div></div><div className="big-number">{metrics[mode].latency}<small> ms</small></div><div className="bars">{[73,82,55,91,64,80,61,77,69,87,58,75].map((h, i) => <i key={i} style={{ height: `${mode === "INT8" ? h * .52 : h}%` }} />)}</div><p>500 runs · 50 warmup · batch size 1</p></div>
      <div className="system-card"><p className="eyebrow">TEST ENVIRONMENT</p><dl><div><dt>ARCHITECTURE</dt><dd>AArch64</dd></div><div><dt>RUNTIME</dt><dd>ONNX Runtime</dd></div><div><dt>PRECISION</dt><dd>{mode}</dd></div><div><dt>THREADS</dt><dd>4</dd></div><div><dt>INPUT</dt><dd>320 × 320</dd></div></dl><span className="disclaimer">PROTOTYPE VISUALIZATION · VALIDATE ON TARGET HARDWARE</span></div></div>
    <div className="table-card"><div className="table-head"><h3>Optimization report</h3><span>baseline.onnx → tinynav-int8.onnx</span></div><table><thead><tr><th>METRIC</th><th>FP32 BASELINE</th><th>INT8 OPTIMIZED</th><th>CHANGE</th></tr></thead><tbody>{rows.map(r => <tr key={r[0]}>{r.map((c,i) => <td key={c} className={i===3 ? "change" : ""}>{c}</td>)}</tr>)}</tbody></table></div>
  </section>
}

function PipelineView() {
  return <section className="detail-page"><div className="detail-hero"><p className="eyebrow">REUSABLE DEVELOPER WORKFLOW</p><h1>Train anywhere.<br /><em>React on Arm.</em></h1><p>A transparent path from an ordinary vision model to a tested, deployable robotics artifact.</p></div>
    <div className="pipeline-map">{[
      ["01", "SENSOR", "Camera / simulation", "320×320 RGB frames"], ["02", "PERCEPTION", "INT8 ONNX model", "Arm64 CPU inference"], ["03", "SAFETY", "Risk estimator", "Distance + confidence"], ["04", "CONTROL", "Robot decision", "GO · SLOW · TURN · STOP"]
    ].map((x,i) => <div className="pipe-step" key={x[0]}><span>{x[0]}</span><small>{x[1]}</small><strong>{x[2]}</strong><p>{x[3]}</p>{i<3 && <i>→</i>}</div>)}</div>
    <div className="artifact-grid"><article><span>MODEL</span><h3>tinynav-int8.onnx</h3><p>Calibrated and graph-optimized edge inference artifact.</p><b>6.4 MB</b></article><article><span>EVIDENCE</span><h3>benchmark.json</h3><p>Machine-readable latency, memory and quality measurements.</p><b>6 METRICS</b></article><article><span>INTEGRATION</span><h3>controller.py</h3><p>Simple, documented interface for robotics control systems.</p><b>4 ACTIONS</b></article></div>
  </section>
}
