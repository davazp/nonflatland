import shaderCode from "./shader.wgsl?raw";

// Half side of the fundamental domain D = [-L, L)².
const L = 1;
const SHIP_SIZE = 0.07;
const TURN_SPEED = 2.6; // rad/s
const MAX_SPEED = 0.9; // world units/s
const ACCEL = 2.5;

const enum Space {
  Plane = 0,
  Torus = 1,
  Klein = 2,
  Pillowcase = 3,
}

// How the generator `a` acts on the strip it crosses into (see the shader).
function flip(space: Space): [number, number] {
  switch (space) {
    case Space.Klein:
      return [1, -1];
    case Space.Pillowcase:
      return [-1, -1];
    default:
      return [1, 1];
  }
}

const enum Camera {
  Fixed = 0,
  Ship = 1,
}

// The ship is an isometry of the plane, p ↦ M p + t, mapping its local frame
// (nose along +y) to the world. M is column-major: [m00, m10, m01, m11].
// It can be orientation-reversing after crossing the Klein bottle's seam.
interface Iso {
  m: [number, number, number, number];
  t: [number, number];
}

const state = {
  space: Space.Torus as Space,
  camera: Camera.Fixed as Camera,
  zoom: 1,
  speed: 0,
  ship: { m: [1, 0, 0, 1], t: [0, -0.3] } as Iso,
};

function resetShip() {
  state.ship = { m: [1, 0, 0, 1], t: [0, -0.3] };
  state.speed = 0;
}

// Bring the ship back into D by composing with a deck transformation.
// Mirrors `reduce` in the shader.
function reduceShip(ship: Iso) {
  if (state.space === Space.Plane) return;
  const [m00, m10, m01, m11] = ship.m;
  let [x, y] = ship.t;
  const n = Math.floor((x + L) / (2 * L));
  x -= 2 * L * n;
  if (Math.abs(n) % 2 === 1) {
    // Compose with diag(f): a reflection on the Klein bottle, a half-turn
    // on the pillowcase.
    const [fx, fy] = flip(state.space);
    x *= fx;
    y *= fy;
    ship.m = [fx * m00, fy * m10, fx * m01, fy * m11];
  }
  y -= 2 * L * Math.floor((y + L) / (2 * L));
  ship.t = [x, y];
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
  state.space = Number(spaceSelect.value) as Space;
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

function update(dt: number) {
  const ship = state.ship;
  const left = keys.has("ArrowLeft") || keys.has("KeyA");
  const right = keys.has("ArrowRight") || keys.has("KeyD");
  const up = keys.has("ArrowUp") || keys.has("KeyW");
  const down = keys.has("ArrowDown") || keys.has("KeyS");

  // Rotate in the ship's own frame: M ← M R(θ). On a mirrored ship this turns
  // the "wrong" way when seen from outside, as it should.
  const theta = ((left ? 1 : 0) - (right ? 1 : 0)) * TURN_SPEED * dt;
  if (theta !== 0) {
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const [m00, m10, m01, m11] = ship.m;
    ship.m = [m00 * c + m01 * s, m10 * c + m11 * s, -m00 * s + m01 * c, -m10 * s + m11 * c];
  }

  const target = up ? MAX_SPEED : down ? -0.5 * MAX_SPEED : 0;
  const dv = target - state.speed;
  state.speed += Math.sign(dv) * Math.min(Math.abs(dv), ACCEL * dt);

  // Forward is M (0, 1).
  ship.t[0] += ship.m[2] * state.speed * dt;
  ship.t[1] += ship.m[3] * state.speed * dt;
  reduceShip(ship);
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

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
    ],
  });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  // Layout matches `Uniforms` in the shader (128 bytes).
  const uniformData = new ArrayBuffer(128);
  const f32 = new Float32Array(uniformData);
  const u32 = new Uint32Array(uniformData);
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // Write an affine map p ↦ M p + t as a mat3x3f (columns padded to vec4).
  function writeMat(offset: number, m: number[], t: number[]) {
    f32.set([m[0], m[1], 0, 0, m[2], m[3], 0, 0, t[0], t[1], 1, 0], offset);
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  let last = performance.now();
  function frame(now: number) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    update(dt);
    resize();

    const { m, t } = state.ship;

    // The only camera-dependent part: the screen → world map.
    if (state.camera === Camera.Fixed) {
      const z = 1.08 * L * state.zoom;
      writeMat(0, [z, 0, 0, z], [0, 0]);
    } else {
      const z = 0.9 * L * state.zoom;
      writeMat(0, m.map((v) => v * z), t);
    }

    // Inverse of the ship isometry: Mᵀ p - Mᵀ t.
    const mt = [m[0], m[2], m[1], m[3]];
    writeMat(12, mt, [-(mt[0] * t[0] + mt[2] * t[1]), -(mt[1] * t[0] + mt[3] * t[1])]);

    f32[24] = canvas.width;
    f32[25] = canvas.height;
    f32[26] = L;
    u32[27] = state.space;
    f32[28] = now / 1000;
    f32[29] = Math.max(0, state.speed / MAX_SPEED);
    u32[30] = state.camera;
    f32[31] = SHIP_SIZE;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
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
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch((e) => showError(String(e)));
