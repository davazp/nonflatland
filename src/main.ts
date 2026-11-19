import shaderCode from "./shader.wgsl?raw";

// All geometry lives in the shader: the metric, geodesics, chart
// transitions and the ship's motion (see shader.wgsl). This file only
// handles input and drives the GPU.

const TURN_SPEED = 2.6; // rad/s
const ACCEL = 2.8; // normalized speed units/s
const FIRE_INTERVAL = 0.15; // s between balls while Space is held

// `Loop` in the shader: a 48-byte header, block bounding boxes, then points
// and scratch space. Only the header is read back.
const LOOP_MAX = 256;
const LOOP_BLOCKS = LOOP_MAX / 16;
const LOOP_HEADER_SIZE = 48;
const LOOP_BUFFER_SIZE = LOOP_HEADER_SIZE + LOOP_BLOCKS * 16 + 2 * LOOP_MAX * 32;

const enum Camera {
  Fixed = 0,
  Ship = 1,
}

const state = {
  space: 1,
  camera: Camera.Fixed as Camera,
  zoom: 1,
  speed: 0, // -0.5 .. 1
  showCurvature: true,
  reset: true,
  loopToggle: false,
  fireCooldown: 0,
};

// `zoom` scales the visible region: larger values show more of the space.
function zoomBy(factor: number) {
  state.zoom = Math.min(6, Math.max(0.25, state.zoom * factor));
}

function resetShip() {
  state.reset = true;
  state.speed = 0;
}

// ---------------------------------------------------------------------------
// Input

const keys = new Set<string>();
const GAME_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Space"]);

const spaceSelect = document.querySelector<HTMLSelectElement>("#space")!;
const cameraSelect = document.querySelector<HTMLSelectElement>("#camera")!;
spaceSelect.value = String(state.space);
cameraSelect.value = String(state.camera);

spaceSelect.addEventListener("change", () => {
  state.space = Number(spaceSelect.value);
  state.zoom = 1;
  resetShip();
  spaceSelect.blur();
});

cameraSelect.addEventListener("change", () => {
  state.camera = Number(cameraSelect.value) as Camera;
  cameraSelect.blur();
});

window.addEventListener(
  "keydown",
  (e) => {
    if (GAME_KEYS.has(e.code)) {
      // Keep arrows from changing a focused <select>.
      e.preventDefault();
      (document.activeElement as HTMLElement | null)?.blur();
    }
    keys.add(e.code);
    if (e.key === "+" || e.key === "=" || e.code === "NumpadAdd") {
      zoomBy(1 / 1.2);
      return;
    } else if (e.key === "-" || e.key === "_" || e.code === "NumpadSubtract") {
      zoomBy(1.2);
      return;
    }
    if (e.repeat) return;
    if (e.code === "KeyC") {
      state.camera = state.camera === Camera.Fixed ? Camera.Ship : Camera.Fixed;
      cameraSelect.value = String(state.camera);
    } else if (e.code === "KeyR") {
      resetShip();
    } else if (e.code === "KeyT") {
      state.loopToggle = true;
    } else if (e.code === "KeyK") {
      state.showCurvature = !state.showCurvature;
    }
  },
  { capture: true },
);
window.addEventListener("keyup", (e) => keys.delete(e.code));
window.addEventListener("blur", () => keys.clear());

window.addEventListener(
  "wheel",
  (e) => {
    zoomBy(Math.exp(e.deltaY * 0.001));
  },
  { passive: true },
);

// Returns the turn angle for this frame and updates the speed.
function readControls(dt: number): number {
  const left = keys.has("ArrowLeft") || keys.has("KeyA");
  const right = keys.has("ArrowRight") || keys.has("KeyD");
  const up = keys.has("ArrowUp") || keys.has("KeyW");
  const down = keys.has("ArrowDown") || keys.has("KeyS");

  const target = up ? 1 : down ? -0.5 : 0;
  const dv = target - state.speed;
  state.speed += Math.sign(dv) * Math.min(Math.abs(dv), ACCEL * dt);

  return ((left ? 1 : 0) - (right ? 1 : 0)) * TURN_SPEED * dt;
}

// Whether to launch a ball this frame: on press, then repeatedly while held.
function readFire(dt: number): boolean {
  state.fireCooldown = Math.max(0, state.fireCooldown - dt);
  if (!keys.has("Space") || state.fireCooldown > 0) return false;
  state.fireCooldown = FIRE_INTERVAL;
  return true;
}

// ---------------------------------------------------------------------------
// Rendering

const loopStatus = document.querySelector<HTMLDivElement>("#loop-status")!;

// Mirrors the LOOP_* modes in the shader.
function showLoopStatus(mode: number, length: number) {
  const len = length.toFixed(2);
  const messages = [
    "T: trace a loop",
    "Tracing… press T to close the loop",
    `Contracting… length ${len}`,
    "Contracted to a point: the loop is trivial in π₁",
    `Stuck at a closed geodesic of length ${len}: the loop is not contractible`,
  ];
  loopStatus.textContent = messages[mode] ?? "";
  loopStatus.dataset.mode = String(mode);
}

function showError(msg: string) {
  const el = document.querySelector<HTMLDivElement>("#error")!;
  el.textContent = msg;
  el.hidden = false;
}

async function main() {
  if (!navigator.gpu) {
    showError("WebGPU is not available in this browser.");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    showError("No WebGPU adapter found.");
    return;
  }
  const device = await adapter.requestDevice();

  const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
  const context = canvas.getContext("webgpu")!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  const module = device.createShaderModule({ code: shaderCode });
  const info = await module.getCompilationInfo();
  for (const m of info.messages) {
    console[m.type === "error" ? "error" : "warn"](`WGSL ${m.lineNum}:${m.linePos} ${m.message}`);
  }

  // Layout matches `Uniforms` in the shader (64 bytes).
  const uniformData = new ArrayBuffer(64);
  const f32 = new Float32Array(uniformData);
  const u32 = new Uint32Array(uniformData);
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // `World` in the shader (ship and balls): written by the compute pass,
  // read when drawing. 48 bytes of header plus 32 balls of 48 bytes.
  const worldBuffer = device.createBuffer({ size: 48 + 32 * 48, usage: GPUBufferUsage.STORAGE });

  // The loop, plus a small buffer to read its header back for the status line.
  const loopBuffer = device.createBuffer({
    size: LOOP_BUFFER_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const loopReadback = device.createBuffer({
    size: LOOP_HEADER_SIZE,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  let readbackPending = false;

  const computeLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const renderLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
    ],
  });

  const computePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [computeLayout] });
  const updatePipeline = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: { module, entryPoint: "update" },
  });
  const contractPipeline = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: { module, entryPoint: "contractLoop" },
  });
  const renderPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renderLayout] }),
    vertex: { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const computeBindGroup = device.createBindGroup({
    layout: computeLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: worldBuffer } },
      { binding: 3, resource: { buffer: loopBuffer } },
    ],
  });
  const renderBindGroup = device.createBindGroup({
    layout: renderLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 2, resource: { buffer: worldBuffer } },
      { binding: 4, resource: { buffer: loopBuffer } },
    ],
  });

  // The ship camera traces a geodesic per pixel, so render at CSS resolution.
  function resize() {
    const w = Math.max(1, Math.round(canvas.clientWidth));
    const h = Math.max(1, Math.round(canvas.clientHeight));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  let last = performance.now();
  function frame(now: number) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const turn = readControls(dt);
    const fire = readFire(dt);
    resize();

    f32[0] = canvas.width;
    f32[1] = canvas.height;
    f32[2] = now / 1000;
    f32[3] = dt;
    u32[4] = state.space;
    u32[5] = state.camera;
    f32[6] = state.zoom;
    f32[7] = state.speed;
    f32[8] = turn;
    u32[9] = state.reset ? 1 : 0;
    u32[10] = state.showCurvature ? 1 : 0;
    u32[11] = fire ? 1 : 0;
    u32[12] = state.loopToggle ? 1 : 0;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);
    state.reset = false;
    state.loopToggle = false;

    const encoder = device.createCommandEncoder();

    const compute = encoder.beginComputePass();
    compute.setBindGroup(0, computeBindGroup);
    compute.setPipeline(updatePipeline);
    compute.dispatchWorkgroups(1);
    compute.setPipeline(contractPipeline);
    compute.dispatchWorkgroups(1);
    compute.end();

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    pass.setPipeline(renderPipeline);
    pass.setBindGroup(0, renderBindGroup);
    pass.draw(3);
    pass.end();

    const readback = !readbackPending;
    if (readback) {
      encoder.copyBufferToBuffer(loopBuffer, 0, loopReadback, 0, LOOP_HEADER_SIZE);
    }
    device.queue.submit([encoder.finish()]);

    if (readback) {
      readbackPending = true;
      loopReadback.mapAsync(GPUMapMode.READ).then(() => {
        const header = loopReadback.getMappedRange();
        showLoopStatus(new Uint32Array(header)[0], new Float32Array(header)[4]);
        loopReadback.unmap();
        readbackPending = false;
      });
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch((e) => showError(String(e)));
