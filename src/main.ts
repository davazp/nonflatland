import shaderCode from "./shader.wgsl?raw";

// All geometry lives in the shader: the metric, geodesics, chart
// transitions and the ship's motion (see shader.wgsl). This file only
// handles input and drives the GPU.

const TURN_SPEED = 2.6; // rad/s
const ACCEL = 2.8; // normalized speed units/s

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
};

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
    if (e.repeat) return;
    if (e.code === "KeyC") {
      state.camera = state.camera === Camera.Fixed ? Camera.Ship : Camera.Fixed;
      cameraSelect.value = String(state.camera);
    } else if (e.code === "KeyR") {
      resetShip();
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
    state.zoom = Math.min(6, Math.max(0.25, state.zoom * Math.exp(e.deltaY * 0.001)));
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

// ---------------------------------------------------------------------------
// Rendering

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

  // Layout matches `Uniforms` in the shader (48 bytes).
  const uniformData = new ArrayBuffer(48);
  const f32 = new Float32Array(uniformData);
  const u32 = new Uint32Array(uniformData);
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // `Ship` in the shader: written by the compute pass, read when drawing.
  const shipBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE });

  const computeLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const renderLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
    ],
  });

  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [computeLayout] }),
    compute: { module, entryPoint: "updateShip" },
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
      { binding: 1, resource: { buffer: shipBuffer } },
    ],
  });
  const renderBindGroup = device.createBindGroup({
    layout: renderLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 2, resource: { buffer: shipBuffer } },
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
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);
    state.reset = false;

    const encoder = device.createCommandEncoder();

    const compute = encoder.beginComputePass();
    compute.setPipeline(computePipeline);
    compute.setBindGroup(0, computeBindGroup);
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

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch((e) => showError(String(e)));
