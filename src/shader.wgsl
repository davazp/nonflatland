// Every space is described by a chart: coordinates x ∈ R² (plus a `sheet`
// index for spaces that need two charts), and
//
//   - metric(x): the Riemannian metric G(x), a 2×2 symmetric matrix giving
//     the inner product of tangent vectors at x. Everything geometric
//     (geodesics, parallel transport, curvature) is derived from it
//     numerically, so any smooth metric works.
//
//   - reduce(x): the chart transitions. It maps a point to its canonical
//     representative and returns the Jacobian of the map, used to carry
//     tangent vectors along. Transitions must be isometries of the metric.
//
// Flat spaces use G = I on the plane with rigid-motion gluings (deck group):
//
//   a(x, y) = (2L + f.x·x, f.y·y),   b(x, y) = (x, y + 2L)
//
//   f = (+1, +1) torus, (+1, -1) Klein bottle, (-1, -1) pillowcase.
//
// The donut and hilly tori use the same square and gluings with a
// non-flat periodic metric. The sphere and projective plane use the
// stereographic metric on the unit disk; the genus-2 (Bolza) surface uses
// the Poincaré disk metric on a regular octagon with opposite sides glued.
//
// The ship is a point plus a frame (e1, e2) orthonormal for G, updated by
// the `step` compute shader: it moves along a geodesic and its frame is
// parallel transported. In the ship camera each pixel shoots a geodesic
// from the ship (the exponential map), which is what a 2D inhabitant
// whose light follows geodesics would see.

struct Uniforms {
  resolution: vec2f,
  time: f32,
  dt: f32,
  space: u32,
  camera: u32,         // 0 fixed (chart view), 1 ship (exponential map)
  zoom: f32,
  speed: f32,          // -0.5 .. 1, scaled per space
  turn: f32,           // radians to turn this frame
  reset: u32,
  showCurvature: u32,
  fire: u32,           // 1 to launch a ball this frame
  loopToggle: u32,     // 1 when T was pressed this frame
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};

struct Ship {
  p: vec2f,
  e1: vec2f,           // right
  e2: vec2f,           // forward
  sheet: f32,
  _pad: f32,
};

// A ball fired by the ship. It moves along a geodesic, like the ship.
struct Ball {
  p: vec2f,
  v: vec2f,            // chart velocity
  g: vec4f,            // metric at p (g11, g12, g22), cached for drawing
  sheet: f32,
  age: f32,            // seconds; dead once >= BALL_LIFE
  _pad: vec2f,
};

const BALL_COUNT = 32;
const BALL_LIFE = 4.0;

struct World {
  ship: Ship,
  next: u32,           // ring buffer index of the next ball to launch
  balls: array<Ball, BALL_COUNT>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> worldRW: World;
@group(0) @binding(2) var<storage, read> worldRO: World;

// A loop traced by the ship, then contracted by curve shortening. Points
// are stored reduced; `next` is the following point expressed in this
// point's chart, so segments can be drawn and measured across seams.
const LOOP_MAX = 256;
const LOOP_BLOCK = 16;
const LOOP_BLOCKS = LOOP_MAX / LOOP_BLOCK;

const LOOP_IDLE = 0u;
const LOOP_RECORDING = 1u;
const LOOP_CONTRACTING = 2u;
const LOOP_CONTRACTED = 3u;   // shrank to a point: trivial in π₁
const LOOP_STUCK = 4u;        // converged to a closed geodesic

struct LoopPoint {
  x: vec2f,
  next: vec2f,
  sheet: f32,
  _pad: f32,
  _pad2: vec2f,
};

struct Loop {
  mode: u32,
  count: u32,
  stable: u32,         // consecutive frames with almost no change in length
  kicks: u32,          // sideways pushes tried since the loop stopped changing
  length: f32,
  age: f32,            // seconds in the current mode
  _pad2: vec2f,
  _pad3: vec4f,
  // Chart bounding boxes (min.xy, max.xy) of the segments of each block of
  // LOOP_BLOCK points, to cull them when drawing.
  blocks: array<vec4f, LOOP_BLOCKS>,
  points: array<LoopPoint, LOOP_MAX>,
  scratch: array<LoopPoint, LOOP_MAX>,
};

@group(0) @binding(3) var<storage, read_write> loopRW: Loop;
@group(0) @binding(4) var<storage, read> loopRO: Loop;

const PI = 3.14159265;

const PLANE = 0u;
const TORUS = 1u;
const KLEIN = 2u;
const PILLOW = 3u;
const SPHERE = 4u;
const RP2 = 5u;
const DONUT = 6u;
const HILLS = 7u;
const BOLZA = 8u;

const DONUT_R = 2.0;
const DONUT_RR = 0.8;

// Regular octagon with 45° angles in the Poincaré disk. Side k is the
// geodesic at distance ρ from the origin in direction kπ/4, where
// cosh ρ = 1 + √2. It is the circle of radius BOLZA_R centred at distance
// BOLZA_C, and is glued to side k+4 by a hyperbolic translation of 2ρ.
const BOLZA_C = 1.0986841;
const BOLZA_R = 0.4550899;
const BOLZA_A = 0.9101797; // tanh ρ

struct Params {
  L: f32,          // size of the canonical domain in chart units
  shipSize: f32,   // intrinsic
  speed: f32,      // intrinsic units / s at full thrust
  fixedScale: f32, // chart units per half screen height, fixed camera
  shipView: f32,   // intrinsic units per half screen height, ship camera
  maxStep: f32,    // intrinsic length of a geodesic integration step
};

fn params() -> Params {
  switch u.space {
    case DONUT: { return Params(PI, 0.2, 2.0, 1.08 * PI, 3.0, 0.08); }
    case HILLS: { return Params(1.5, 0.08, 1.0, 1.08 * 1.5, 1.3, 0.05); }
    case SPHERE: { return Params(1.0, 0.1, 0.8, 1.45, 2.0, 0.05); }
    case RP2: { return Params(1.0, 0.1, 0.8, 1.15, 2.0, 0.05); }
    case BOLZA: { return Params(0.6, 0.12, 0.9, 1.05, 2.2, 0.05); }
    default: { return Params(1.0, 0.07, 0.9, 1.08, 0.9, 1e3); }
  }
}

fn isFlat() -> bool {
  return u.space <= PILLOW;
}

fn isSquare() -> bool {
  return u.space <= PILLOW || u.space == DONUT || u.space == HILLS;
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Complex numbers and Möbius maps (for the Poincaré disk)

fn cmul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cdiv(a: vec2f, b: vec2f) -> vec2f {
  return cmul(a, vec2f(b.x, -b.y)) / dot(b, b);
}

// Hyperbolic translation z ↦ (z + a) / (1 + ā z), mapping 0 to a.
fn mobius(a: vec2f, z: vec2f) -> vec2f {
  return cdiv(z + a, vec2f(1.0, 0.0) + cmul(vec2f(a.x, -a.y), z));
}

// Its derivative, as the 2×2 matrix of a complex multiplication.
fn mobiusJ(a: vec2f, z: vec2f) -> mat2x2f {
  let d = vec2f(1.0, 0.0) + cmul(vec2f(a.x, -a.y), z);
  let w = cdiv(vec2f(1.0 - dot(a, a), 0.0), cmul(d, d));
  return mat2x2f(w.x, w.y, -w.y, w.x);
}

fn bolzaDir(k: i32) -> vec2f {
  let t = f32(k) * PI / 4.0;
  return vec2f(cos(t), sin(t));
}

// Signed distance (in chart units) to the circle of side k; negative means
// beyond the side, i.e. outside the octagon.
fn bolzaSide(k: i32, z: vec2f) -> f32 {
  return length(z - BOLZA_C * bolzaDir(k)) - BOLZA_R;
}

// ---------------------------------------------------------------------------
// The metric

// Height function of the hilly torus and its gradient: (h, ∂x h, ∂y h).
// Periodic bumps exp((cos θx + cos θy - 2) / w²) with θ = π (x - c) / L.
fn bump(x: vec2f, c: vec2f, a: f32, w: f32) -> vec3f {
  let k = PI / 1.5;
  let t = k * (x - c);
  let e = a * exp((cos(t.x) + cos(t.y) - 2.0) / (w * w));
  return vec3f(e, -e * k * sin(t.x) / (w * w), -e * k * sin(t.y) / (w * w));
}

fn hills(x: vec2f) -> vec3f {
  return bump(x, vec2f(0.65, 0.7), 0.35, 0.5)
       + bump(x, vec2f(-0.75, -0.65), -0.35, 0.5)
       + bump(x, vec2f(1.0, -1.1), 0.2, 0.45);
}

// G(x) as (g11, g12, g22).
fn metric(x: vec2f) -> vec3f {
  switch u.space {
    case SPHERE, RP2: {
      let l = 2.0 / (1.0 + dot(x, x));
      return vec3f(l * l, 0.0, l * l);
    }
    case BOLZA: {
      let l = 2.0 / (1.0 - min(dot(x, x), 0.999));
      return vec3f(l * l, 0.0, l * l);
    }
    case DONUT: {
      let a = DONUT_R + DONUT_RR * cos(x.y);
      return vec3f(a * a, 0.0, DONUT_RR * DONUT_RR);
    }
    case HILLS: {
      // Induced metric of the graph z = h(x, y): I + ∇h ∇hᵀ.
      let g = hills(x).yz;
      return vec3f(1.0 + g.x * g.x, g.x * g.y, 1.0 + g.y * g.y);
    }
    default: {
      return vec3f(1.0, 0.0, 1.0);
    }
  }
}

fn metricMat(x: vec2f) -> mat2x2f {
  let g = metric(x);
  return mat2x2f(g.x, g.y, g.y, g.z);
}

fn metricDet(x: vec2f) -> f32 {
  let g = metric(x);
  return g.x * g.z - g.y * g.y;
}

// Christoffel symbols of the Levi-Civita connection, Γ^k_ij as two
// symmetric matrices (k = 1, 2), from finite differences of the metric:
//   Γ^k_ij = ½ g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
struct Gamma {
  g1: mat2x2f,
  g2: mat2x2f,
};

fn christoffel(x: vec2f) -> Gamma {
  if (isFlat()) {
    return Gamma(mat2x2f(), mat2x2f());
  }
  let h = 1e-3;
  let g = metric(x);
  let dx = (metric(x + vec2f(h, 0.0)) - metric(x - vec2f(h, 0.0))) / (2.0 * h);
  let dy = (metric(x + vec2f(0.0, h)) - metric(x - vec2f(0.0, h))) / (2.0 * h);
  // Lowered symbols Γ_l,ij as (11, 12, 22).
  let l1 = vec3f(0.5 * dx.x, 0.5 * dy.x, dy.y - 0.5 * dx.z);
  let l2 = vec3f(dx.y - 0.5 * dy.x, 0.5 * dx.z, 0.5 * dy.z);
  let det = g.x * g.z - g.y * g.y;
  let u1 = (g.z * l1 - g.y * l2) / det;
  let u2 = (-g.y * l1 + g.x * l2) / det;
  return Gamma(mat2x2f(u1.x, u1.y, u1.y, u1.z), mat2x2f(u2.x, u2.y, u2.y, u2.z));
}

// -Γ(a, b): the covariant correction for a vector b moved along a.
fn connection(c: Gamma, a: vec2f, b: vec2f) -> vec2f {
  return -vec2f(dot(a, c.g1 * b), dot(a, c.g2 * b));
}

// Gaussian curvature from derivatives of the Christoffel symbols.
fn curvature(x: vec2f) -> f32 {
  if (isFlat()) {
    return 0.0;
  }
  let h = 1e-2;
  let c = christoffel(x);
  let xp = christoffel(x + vec2f(h, 0.0));
  let xm = christoffel(x - vec2f(h, 0.0));
  let yp = christoffel(x + vec2f(0.0, h));
  let ym = christoffel(x - vec2f(0.0, h));
  // Q^l = ∂1 Γ^l_22 - ∂2 Γ^l_12 + Γ^l_1m Γ^m_22 - Γ^l_2m Γ^m_12
  let q1 = (xp.g1[1][1] - xm.g1[1][1] - yp.g1[0][1] + ym.g1[0][1]) / (2.0 * h)
         + c.g1[0][0] * c.g1[1][1] + c.g1[0][1] * c.g2[1][1]
         - c.g1[1][0] * c.g1[0][1] - c.g1[1][1] * c.g2[0][1];
  let q2 = (xp.g2[1][1] - xm.g2[1][1] - yp.g2[0][1] + ym.g2[0][1]) / (2.0 * h)
         + c.g2[0][0] * c.g1[1][1] + c.g2[0][1] * c.g2[1][1]
         - c.g2[1][0] * c.g1[0][1] - c.g2[1][1] * c.g2[0][1];
  let g = metric(x);
  return (g.x * q1 + g.y * q2) / (g.x * g.z - g.y * g.y);
}

// ---------------------------------------------------------------------------
// Chart transitions

struct Chart {
  x: vec2f,
  sheet: f32,
  J: mat2x2f,   // Jacobian of the transition, for tangent vectors
};

fn coverFlip() -> vec2f {
  switch u.space {
    case KLEIN: { return vec2f(1.0, -1.0); }
    case PILLOW: { return vec2f(-1.0, -1.0); }
    default: { return vec2f(1.0, 1.0); }
  }
}

// Inversion in the unit circle and its Jacobian.
fn inversionJ(x: vec2f) -> mat2x2f {
  let r2 = dot(x, x);
  return (mat2x2f(1.0, 0.0, 0.0, 1.0) * r2 - 2.0 * mat2x2f(x * x.x, x * x.y)) * (1.0 / (r2 * r2));
}

// Map a point to its representative in the canonical domain.
fn reduce(x: vec2f, sheet: f32) -> Chart {
  let I = mat2x2f(1.0, 0.0, 0.0, 1.0);
  switch u.space {
    case PLANE: {
      return Chart(x, sheet, I);
    }
    case SPHERE: {
      // Unit disk of the north/south stereographic charts. Leaving one
      // means entering the other, seen from outside: (x, y) ↦ (-x, y) / r².
      let r2 = dot(x, x);
      if (r2 <= 1.0) {
        return Chart(x, sheet, I);
      }
      let D = mat2x2f(-1.0, 0.0, 0.0, 1.0);
      return Chart(vec2f(-x.x, x.y) / r2, 1.0 - sheet, D * inversionJ(x));
    }
    case RP2: {
      // Sphere modulo the antipodal map, which is x ↦ -x / r² here.
      let r2 = dot(x, x);
      if (r2 <= 1.0) {
        return Chart(x, sheet, I);
      }
      return Chart(-x / r2, sheet, -1.0 * inversionJ(x));
    }
    case BOLZA: {
      var z = x;
      var J = I;
      for (var i = 0; i < 12; i++) {
        var moved = false;
        for (var k = 0; k < 8; k++) {
          if (bolzaSide(k, z) < 0.0) {
            let a = -BOLZA_A * bolzaDir(k);
            J = mobiusJ(a, z) * J;
            z = mobius(a, z);
            moved = true;
            break;
          }
        }
        if (!moved) {
          break;
        }
      }
      return Chart(z, sheet, J);
    }
    default: {
      let L = params().L;
      var p = x;
      var J = I;
      let n = floor((p.x + L) / (2.0 * L));
      p.x -= n * 2.0 * L;
      if ((i32(n) & 1) != 0) {
        let f = coverFlip();
        p *= f;
        J = mat2x2f(f.x, 0.0, 0.0, f.y);
      }
      p.y -= floor((p.y + L) / (2.0 * L)) * 2.0 * L;
      return Chart(p, sheet, J);
    }
  }
}

fn inDomain(x: vec2f) -> bool {
  switch u.space {
    case PLANE: { return true; }
    case SPHERE, RP2: { return dot(x, x) <= 1.0; }
    case BOLZA: {
      for (var k = 0; k < 8; k++) {
        if (bolzaSide(k, x) < 0.0) {
          return false;
        }
      }
      return true;
    }
    default: { return all(abs(x) <= vec2f(params().L)); }
  }
}

// Images of a canonical point q in the neighbouring copies of the domain,
// used to draw the ship when it straddles an edge: (x, sheet).
fn liftCount() -> i32 {
  switch u.space {
    case PLANE: { return 1; }
    case SPHERE, RP2: { return 2; }
    default: { return 9; }
  }
}

fn lift(k: i32, q: vec2f, sheet: f32) -> vec3f {
  if (k == 0 && !isSquare()) {
    return vec3f(q, sheet);
  }
  switch u.space {
    case SPHERE: {
      return vec3f(vec2f(-q.x, q.y) / max(dot(q, q), 1e-6), 1.0 - sheet);
    }
    case RP2: {
      return vec3f(-q / max(dot(q, q), 1e-6), sheet);
    }
    case BOLZA: {
      return vec3f(mobius(BOLZA_A * bolzaDir(k - 1), q), sheet);
    }
    default: {
      // The 3×3 block of copies around the domain; k = 4 is the identity.
      if (u.space == PLANE) {
        return vec3f(q, sheet);
      }
      let L = params().L;
      let i = k % 3 - 1;
      let j = k / 3 - 1;
      let f = select(vec2f(1.0), coverFlip(), (i & 1) != 0);
      return vec3f(2.0 * L * vec2f(f32(i), f32(j)) + f * q, sheet);
    }
  }
}

// ---------------------------------------------------------------------------
// Geodesics

struct Geo {
  x: vec2f,
  v: vec2f,
  sheet: f32,
};

fn geoStep(g: Geo, h: f32) -> Geo {
  let k1x = g.v;
  let k1v = connection(christoffel(g.x), g.v, g.v);
  let v2 = g.v + 0.5 * h * k1v;
  let k2v = connection(christoffel(g.x + 0.5 * h * k1x), v2, v2);
  let v3 = g.v + 0.5 * h * k2v;
  let k3v = connection(christoffel(g.x + 0.5 * h * v2), v3, v3);
  let v4 = g.v + h * k3v;
  let k4v = connection(christoffel(g.x + h * v3), v4, v4);
  let x = g.x + h / 6.0 * (k1x + 2.0 * v2 + 2.0 * v3 + v4);
  let v = g.v + h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
  let c = reduce(x, g.sheet);
  return Geo(c.x, c.J * v, c.sheet);
}

// Follow the geodesic from the ship with initial velocity `lp`, given in
// the ship's orthonormal frame, for unit time.
fn exponential(lp: vec2f) -> Geo {
  let ship = worldRO.ship;
  var g = Geo(ship.p, ship.e1 * lp.x + ship.e2 * lp.y, ship.sheet);
  let n = clamp(i32(ceil(length(lp) / params().maxStep)), 1, 160);
  let h = 1.0 / f32(n);
  for (var i = 0; i < n; i++) {
    g = geoStep(g, h);
  }
  return g;
}

// ---------------------------------------------------------------------------
// Ship update: move along a geodesic, parallel transporting the frame.

struct Frame {
  x: vec2f,
  e1: vec2f,
  e2: vec2f,
};

fn frameDeriv(f: Frame, d: f32) -> Frame {
  let c = christoffel(f.x);
  let v = d * f.e2;
  return Frame(v, connection(c, v, f.e1), connection(c, v, f.e2));
}

fn frameAdd(a: Frame, b: Frame, h: f32) -> Frame {
  return Frame(a.x + h * b.x, a.e1 + h * b.e1, a.e2 + h * b.e2);
}

fn gdot(G: mat2x2f, a: vec2f, b: vec2f) -> f32 {
  return dot(a, G * b);
}

@compute @workgroup_size(1)
fn update() {
  if (u.reset != 0u) {
    worldRW.ship = Ship(vec2f(0.0, -0.3 * params().L), vec2f(1.0, 0.0), vec2f(0.0, 1.0), 0.0, 0.0);
    for (var i = 0; i < BALL_COUNT; i++) {
      worldRW.balls[i].age = BALL_LIFE;
    }
  }
  updateShip();
  if (u.fire != 0u) {
    launchBall();
  }
  updateBalls();
  updateLoop();
}

fn updateShip() {
  let s = worldRW.ship;

  // Turn within the tangent plane.
  let c = cos(u.turn);
  let sn = sin(u.turn);
  var f = Frame(s.p, c * s.e1 + sn * s.e2, -sn * s.e1 + c * s.e2);
  var sheet = s.sheet;

  // Move forward by d along the geodesic (RK4, a few substeps).
  let n = 4;
  let d = u.speed * params().speed * u.dt / f32(n);
  for (var i = 0; i < n; i++) {
    let k1 = frameDeriv(f, d);
    let k2 = frameDeriv(frameAdd(f, k1, 0.5), d);
    let k3 = frameDeriv(frameAdd(f, k2, 0.5), d);
    let k4 = frameDeriv(frameAdd(f, k3, 1.0), d);
    f = Frame(
      f.x + (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x) / 6.0,
      f.e1 + (k1.e1 + 2.0 * k2.e1 + 2.0 * k3.e1 + k4.e1) / 6.0,
      f.e2 + (k1.e2 + 2.0 * k2.e2 + 2.0 * k3.e2 + k4.e2) / 6.0,
    );
    let ch = reduce(f.x, sheet);
    f = Frame(ch.x, ch.J * f.e1, ch.J * f.e2);
    sheet = ch.sheet;
  }

  // Re-orthonormalize with respect to G (Gram–Schmidt, keeping forward).
  let G = metricMat(f.x);
  let e2 = f.e2 / sqrt(gdot(G, f.e2, f.e2));
  var e1 = f.e1 - gdot(G, f.e1, e2) * e2;
  e1 /= sqrt(gdot(G, e1, e1));

  worldRW.ship = Ship(f.x, e1, e2, sheet, 0.0);
}

// Launch a ball from the nose of the ship, straight ahead.
fn launchBall() {
  let P = params();
  let s = worldRW.ship;
  // Follow the geodesic from the centre of the ship to its nose. Geodesics
  // keep their speed, so the velocity there still has length shipSize.
  let nose = geoStep(Geo(s.p, s.e2 * P.shipSize, s.sheet), 1.0);
  let speed = P.speed * (1.6 + max(u.speed, 0.0));
  let v = nose.v / P.shipSize * speed;

  let i = worldRW.next % u32(BALL_COUNT);
  worldRW.balls[i] = Ball(nose.x, v, vec4f(metric(nose.x), 0.0), nose.sheet, 0.0, vec2f(0.0));
  worldRW.next = (i + 1u) % u32(BALL_COUNT);
}

fn updateBalls() {
  for (var i = 0; i < BALL_COUNT; i++) {
    var b = worldRW.balls[i];
    if (b.age >= BALL_LIFE) {
      continue;
    }
    var g = Geo(b.p, b.v, b.sheet);
    g = geoStep(g, 0.5 * u.dt);
    g = geoStep(g, 0.5 * u.dt);
    worldRW.balls[i] = Ball(g.x, g.v, vec4f(metric(g.x), 0.0), g.sheet, b.age + u.dt, vec2f(0.0));
  }
}

// ---------------------------------------------------------------------------
// Loops and curve shortening
//
// The loop is contracted by gradient descent on its discrete energy
// Σ |x_{i+1} - x_i|²_G. A contractible loop shrinks to a point. Otherwise
// it converges to the shortest closed geodesic in its free homotopy class
// (Birkhoff curve shortening) and stops: the loop is non-trivial in π₁.

// y (canonical, on sheet ys) expressed in the chart of x: its nearest copy.
fn relative(x: vec2f, xs: f32, y: vec2f, ys: f32) -> vec2f {
  var best = y;
  var bestD = 1e30;
  for (var k = 0; k < liftCount(); k++) {
    let l = lift(k, y, ys);
    if (abs(l.z - xs) > 0.5) {
      continue;
    }
    let d = dot(l.xy - x, l.xy - x);
    if (d < bestD) {
      bestD = d;
      best = l.xy;
    }
  }
  return best;
}

// Intrinsic length of a short chart segment.
fn segLength(a: vec2f, b: vec2f) -> f32 {
  let d = b - a;
  return sqrt(max(dot(d, metricMat(0.5 * (a + b)) * d), 0.0));
}

fn segEnergy(a: vec2f, b: vec2f) -> f32 {
  let d = b - a;
  return dot(d, metricMat(0.5 * (a + b)) * d);
}

fn loopSpacing() -> f32 {
  return 0.5 * params().shipSize;
}

fn loopPoint(x: vec2f, sheet: f32) -> LoopPoint {
  return LoopPoint(x, x, sheet, 0.0, vec2f(0.0));
}

fn loopNeighbour(i: u32, offset: u32) -> LoopPoint {
  return loopRW.points[(i + offset) % loopRW.count];
}

fn appendLoopPoint(x: vec2f, sheet: f32) {
  if (loopRW.count < u32(LOOP_MAX)) {
    loopRW.points[loopRW.count] = loopPoint(x, sheet);
    loopRW.count += 1u;
  }
}

// Recompute `next`, the total length and the block bounding boxes.
fn finishLoop(closed: bool) {
  let n = loopRW.count;
  var length = 0.0;
  var lo = vec2f(1e30);
  var hi = vec2f(-1e30);
  for (var i = 0u; i < n; i++) {
    if (i % u32(LOOP_BLOCK) == 0u) {
      lo = vec2f(1e30);
      hi = vec2f(-1e30);
    }
    let p = loopRW.points[i];
    var next = p.next;
    if (closed || i + 1u < n) {
      let q = loopNeighbour(i, 1u);
      next = relative(p.x, p.sheet, q.x, q.sheet);
    }
    loopRW.points[i].next = next;
    length += segLength(p.x, next);
    lo = min(lo, min(p.x, next));
    hi = max(hi, max(p.x, next));
    loopRW.blocks[i / u32(LOOP_BLOCK)] = vec4f(lo, hi);
  }
  loopRW.length = length;
}

// Join the end of the trace to the nearest copy of its start with a chart
// segment. The shortening flow straightens it afterwards; the copy chosen
// decides the homotopy class.
fn closeLoop() {
  let n = loopRW.count;
  let last = loopRW.points[n - 1u];
  let first = loopRW.points[0];
  let start = relative(last.x, last.sheet, first.x, first.sheet);
  let m = max(1, i32(ceil(segLength(last.x, start) / loopSpacing())));
  for (var j = 1; j < m; j++) {
    let c = reduce(mix(last.x, start, f32(j) / f32(m)), last.sheet);
    appendLoopPoint(c.x, c.sheet);
  }
}

fn hash(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
}

// Point i after one step of Riemannian gradient descent on the energy.
fn shortenPoint(i: u32, n: u32) -> LoopPoint {
  let h = 1e-3 * params().L;
  let p = loopRW.points[i];
  let pa = loopNeighbour(i, n - 1u);
  let pb = loopNeighbour(i, 1u);
  let a = relative(p.x, p.sheet, pa.x, pa.sheet);
  let b = relative(p.x, p.sheet, pb.x, pb.sheet);
  let ex = vec2f(h, 0.0);
  let ey = vec2f(0.0, h);
  let grad = vec2f(
    segEnergy(a, p.x + ex) + segEnergy(p.x + ex, b) - segEnergy(a, p.x - ex) - segEnergy(p.x - ex, b),
    segEnergy(a, p.x + ey) + segEnergy(p.x + ey, b) - segEnergy(a, p.x - ey) - segEnergy(p.x - ey, b),
  ) / (2.0 * h);
  let g = metric(p.x);
  let Ginv = mat2x2f(g.z, -g.y, -g.y, g.x) * (1.0 / (g.x * g.z - g.y * g.y));
  // A step of 1/4 would jump to the midpoint in flat space; stay below.
  var step = -0.2 * (Ginv * grad);
  // A tiny sideways jitter so loops don't rest on unstable geodesics,
  // such as a great circle of the sphere.
  let side = vec2f(a.y - b.y, b.x - a.x);
  let sideLen = sqrt(max(dot(side, metricMat(p.x) * side), 1e-12));
  step += side / sideLen * (hash(vec2f(f32(i), u.time)) - 0.5) * 1e-3 * loopSpacing();
  let c = reduce(p.x + step, p.sheet);
  return loopPoint(c.x, c.sheet);
}

// Keep points roughly `spacing` apart: merge close ones (keeping at least
// three) and split long segments.
fn resampleLoop() {
  let n = loopRW.count;
  let spacing = loopSpacing();
  var m = 0u;
  for (var i = 0u; i < n; i++) {
    let p = loopRW.points[i];
    if (m > 0u && m + (n - i) > 3u) {
      let last = loopRW.scratch[m - 1u];
      let d = segLength(last.x, relative(last.x, last.sheet, p.x, p.sheet));
      if (d < 0.5 * spacing) {
        continue;
      }
    }
    loopRW.scratch[m] = p;
    m += 1u;
    let q = loopNeighbour(i, 1u);
    let qn = relative(p.x, p.sheet, q.x, q.sheet);
    if (segLength(p.x, qn) > 1.6 * spacing && m + (n - i) < u32(LOOP_MAX)) {
      let c = reduce(0.5 * (p.x + qn), p.sheet);
      loopRW.scratch[m] = loopPoint(c.x, c.sheet);
      m += 1u;
    }
  }
  for (var i = 0u; i < m; i++) {
    loopRW.points[i] = loopRW.scratch[i];
  }
  loopRW.count = m;
}

fn setLoopMode(mode: u32) {
  loopRW.mode = mode;
  loopRW.age = 0.0;
  loopRW.stable = 0u;
  loopRW.kicks = 0u;
}

// Push every point sideways by a quarter of the spacing. A loop resting on an
// unstable closed geodesic (a great circle, the donut's outer equator)
// gets shorter and keeps contracting; a true minimum comes back.
fn kickLoop(side: f32) {
  let n = loopRW.count;
  for (var i = 0u; i < n; i++) {
    let p = loopRW.points[i];
    let pa = loopNeighbour(i, n - 1u);
    let pb = loopNeighbour(i, 1u);
    let t = relative(p.x, p.sheet, pb.x, pb.sheet) - relative(p.x, p.sheet, pa.x, pa.sheet);
    let nrm = vec2f(-t.y, t.x);
    let len = sqrt(max(dot(nrm, metricMat(p.x) * nrm), 1e-12));
    let c = reduce(p.x + side * 0.25 * loopSpacing() * nrm / len, p.sheet);
    loopRW.points[i] = loopPoint(c.x, c.sheet);
  }
}

fn updateLoop() {
  let ship = worldRW.ship;
  if (u.reset != 0u) {
    setLoopMode(LOOP_IDLE);
    loopRW.count = 0u;
  }
  loopRW.age += u.dt;

  if (u.loopToggle != 0u) {
    if (loopRW.mode == LOOP_RECORDING && loopRW.count >= 2u) {
      closeLoop();
      setLoopMode(LOOP_CONTRACTING);
    } else {
      loopRW.count = 0u;
      appendLoopPoint(ship.p, ship.sheet);
      setLoopMode(LOOP_RECORDING);
    }
  }

  switch loopRW.mode {
    case LOOP_RECORDING: {
      let n = loopRW.count;
      let last = loopRW.points[n - 1u];
      let s = relative(last.x, last.sheet, ship.p, ship.sheet);
      if (segLength(last.x, s) >= loopSpacing()) {
        appendLoopPoint(ship.p, ship.sheet);
        if (loopRW.count == u32(LOOP_MAX)) {
          closeLoop();
          setLoopMode(LOOP_CONTRACTING);
          finishLoop(true);
          return;
        }
      }
      // The open end follows the ship.
      let end = loopRW.points[loopRW.count - 1u];
      loopRW.points[loopRW.count - 1u].next = relative(end.x, end.sheet, ship.p, ship.sheet);
      finishLoop(false);
    }
    default: {}
  }
}

var<workgroup> wgMode: u32;
var<workgroup> wgCount: u32;

// Contract the loop: one invocation per point, Jacobi sweeps between
// barriers, then serial resampling and bookkeeping by the first one.
@compute @workgroup_size(LOOP_MAX)
fn contractLoop(@builtin(local_invocation_index) i: u32) {
  if (i == 0u) {
    wgMode = loopRW.mode;
    wgCount = loopRW.count;
  }
  let mode = workgroupUniformLoad(&wgMode);
  let n = workgroupUniformLoad(&wgCount);
  if (mode != LOOP_CONTRACTING) {
    return;
  }

  for (var it = 0; it < 16; it++) {
    var p: LoopPoint;
    if (i < n) {
      p = shortenPoint(i, n);
    }
    storageBarrier();
    if (i < n) {
      loopRW.points[i] = p;
    }
    storageBarrier();
  }

  if (i == 0u) {
    let before = loopRW.length;
    resampleLoop();
    finishLoop(true);
    let after = loopRW.length;
    if (after < 0.3 * loopSpacing()) {
      setLoopMode(LOOP_CONTRACTED);
    } else if (abs(before - after) < 2e-5 * after) {
      loopRW.stable += 1u;
      if (loopRW.stable > 60u) {
        if (loopRW.kicks < 2u) {
          kickLoop(select(-1.0, 1.0, loopRW.kicks == 0u));
          finishLoop(true);
          loopRW.kicks += 1u;
          loopRW.stable = 0u;
        } else {
          loopRW.mode = LOOP_STUCK;
          loopRW.age = 0.0;
        }
      }
    } else {
      loopRW.stable = 0u;
    }
  }
}

// ---------------------------------------------------------------------------
// SDF helpers

fn sdBox(p: vec2f, c: vec2f, h: vec2f) -> f32 {
  let d = abs(p - c) - h;
  return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0);
}

fn sdSegment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}

fn sdTriangle(p: vec2f, p0: vec2f, p1: vec2f, p2: vec2f) -> f32 {
  let e0 = p1 - p0;
  let e1 = p2 - p1;
  let e2 = p0 - p2;
  let v0 = p - p0;
  let v1 = p - p1;
  let v2 = p - p2;
  let pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
  let pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
  let pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
  let s = sign(e0.x * e2.y - e0.y * e2.x);
  let d = min(min(vec2f(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)),
                  vec2f(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x))),
                  vec2f(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)));
  return -sqrt(d.x) * sign(d.y);
}

fn fill(d: f32, pw: f32) -> f32 {
  return 1.0 - smoothstep(-pw, pw, d);
}

// Distance to the nearest multiple of `spacing`.
fn periodicDist(t: f32, spacing: f32) -> f32 {
  return abs(fract(t / spacing + 0.5) - 0.5) * spacing;
}

// ---------------------------------------------------------------------------
// Background, as a function on the canonical domain (chart coordinates).

fn gridDist(q: vec2f, spacing: f32) -> f32 {
  let g = abs(fract(q / spacing + 0.5) - 0.5) * spacing;
  return min(g.x, g.y);
}

// "^"-shaped chevrons along an axis; `a` runs along, `c` across.
fn chevrons(a: f32, c: f32, period: f32, h: f32) -> f32 {
  let s = a - (floor(a / period) + 0.5) * period;
  let d = abs(s + 0.8 * abs(c)) * 0.78;
  return max(d, abs(c) - h);
}

fn glyphF(p: vec2f) -> f32 {
  var d = sdBox(p, vec2f(-0.15, 0.0), vec2f(0.06, 0.30));
  d = min(d, sdBox(p, vec2f(0.03, 0.24), vec2f(0.18, 0.06)));
  d = min(d, sdBox(p, vec2f(-0.01, 0.01), vec2f(0.14, 0.05)));
  return d;
}

const GRID_MINOR = vec3f(0.22, 0.25, 0.4);
const GRID_MAJOR = vec3f(0.35, 0.4, 0.65);
const CYAN = vec3f(0.3, 0.85, 0.95);
const MAGENTA = vec3f(0.95, 0.35, 0.85);
const YELLOW = vec3f(0.95, 0.85, 0.3);

// Line at distance d (chart units), with a width in pixels.
fn line(col: vec3f, c: vec3f, d: f32, width: f32, pw: f32, a: f32) -> vec3f {
  return mix(col, c, a * fill(d - width * pw, pw));
}

fn squarePattern(q: vec2f, pw: f32) -> vec3f {
  let L = params().L;
  let k = PI / L;

  // Smooth background, periodic and even in y so it is well defined
  // on the torus, the Klein bottle and the pillowcase.
  var col = mix(vec3f(0.025, 0.03, 0.07), vec3f(0.08, 0.035, 0.11), 0.5 + 0.5 * cos(k * q.x));
  col += vec3f(0.0, 0.04, 0.06) * (0.5 + 0.5 * cos(k * q.y));

  col = line(col, GRID_MINOR, gridDist(q, L / 8.0), 0.25, pw, 0.35);
  col = line(col, GRID_MAJOR, gridDist(q, L / 2.0), 0.6, pw, 0.6);

  if (u.space == HILLS) {
    // Contour lines of the height function.
    let h = hills(q);
    let d = periodicDist(h.x, 0.05) / max(length(h.yz), 1e-3);
    col = line(col, vec3f(0.5, 0.9, 0.5), d, 0.4, pw, 0.45);
  }

  // Seams of the square, with the classic gluing-diagram arrows.
  if (u.space != PLANE) {
    let edge = L - abs(q);
    let inset = 0.07 * L;
    let period = 0.25 * L;
    let h = 0.03 * L;
    var xSeamColor = CYAN;
    if (u.space == KLEIN) {
      xSeamColor = MAGENTA;
    } else if (u.space == PILLOW) {
      xSeamColor = YELLOW;
    }

    // Bottom/top edges: glued by b, arrows point +x on both.
    let cy = chevrons(q.x, edge.y - inset, period, h) - 0.25 * pw;
    col = mix(col, CYAN, 0.8 * fill(cy, pw));
    col = line(col, CYAN, edge.y, 0.8, pw, 0.5);

    // Left/right edges. Torus and Klein bottle glue them to each other with
    // a, the Klein bottle reversing direction. The pillowcase folds each one
    // onto itself about its midpoint, so arrows point away from it.
    var along = q.y;
    if (u.space == PILLOW) {
      along = abs(q.y);
    } else if (q.x > 0.0) {
      along *= coverFlip().y;
    }
    let cx = chevrons(along, edge.x - inset, period, h) - 0.25 * pw;
    col = mix(col, xSeamColor, 0.8 * fill(cx, pw));
    col = line(col, xSeamColor, edge.x, 0.8, pw, 0.5);

    // Cone points of the pillowcase.
    if (u.space == PILLOW) {
      let aq = abs(q);
      let dc = min(length(aq - vec2f(L, 0.0)), length(aq - vec2f(L, L)));
      col = mix(col, xSeamColor, fill(abs(dc - 0.06 * L) - 0.008 * L, pw));
      col = mix(col, vec3f(1.0), fill(dc - 0.02 * L, pw));
    }
  }
  return col;
}

// Latitude/longitude lines of the round sphere in stereographic charts.
fn spherePattern(q: vec2f, sheet: f32, pw: f32) -> vec3f {
  let r = length(q);
  let a = 2.0 * atan(r);
  let colat = select(a, PI - a, sheet > 0.5);
  var col = mix(vec3f(0.09, 0.04, 0.10), vec3f(0.02, 0.05, 0.10), colat / PI);

  let dLat = periodicDist(colat, PI / 12.0) * (1.0 + r * r) / 2.0;
  let dLon = periodicDist(atan2(q.y, q.x), PI / 12.0) * r;
  let lonFade = smoothstep(0.1, 0.3, r);
  col = line(col, GRID_MAJOR, dLat, 0.5, pw, 0.5);
  col = line(col, GRID_MAJOR, dLon, 0.5, pw, 0.5 * lonFade);

  // Chart boundary: the equator. On the projective plane it is glued to
  // itself antipodally.
  let seam = select(CYAN, MAGENTA, u.space == RP2);
  col = line(col, seam, abs(1.0 - r) * 2.0, 0.8, pw, 0.7);
  return col;
}

// Geodesic polar grid and the octagon of the genus-2 surface.
fn bolzaPattern(q: vec2f, pw: f32) -> vec3f {
  let r = length(q);
  let dist = log((1.0 + r) / (1.0 - r));
  var col = mix(vec3f(0.03, 0.06, 0.09), vec3f(0.09, 0.03, 0.09), smoothstep(0.0, 2.4, dist));

  let dCirc = periodicDist(dist, 0.5) * (1.0 - r * r) / 2.0;
  let dRad = periodicDist(atan2(q.y, q.x), PI / 8.0) * r;
  col = line(col, GRID_MINOR, dCirc, 0.4, pw, 0.6);
  col = line(col, GRID_MINOR, dRad, 0.4, pw, 0.6 * smoothstep(0.03, 0.1, r));

  // Sides, coloured by pair (k and k+4 are glued).
  var colors = array<vec3f, 4>(CYAN, MAGENTA, YELLOW, vec3f(0.5, 0.95, 0.5));
  for (var k = 0; k < 8; k++) {
    col = line(col, colors[k % 4], abs(bolzaSide(k, q)), 0.9, pw, 0.8);
  }
  return col;
}

// Asymmetric decorations: they reveal reflections and rotations.
fn decorations(col0: vec3f, q: vec2f, pw: f32) -> vec3f {
  var col = col0;
  let L = params().L;

  // Origin frame: red +x, green +y
  let fx = sdSegment(q, vec2f(0.0), vec2f(0.22 * L, 0.0)) - 0.012 * L;
  let fy = sdSegment(q, vec2f(0.0), vec2f(0.0, 0.22 * L)) - 0.012 * L;
  col = mix(col, vec3f(0.95, 0.3, 0.3), fill(fx, pw));
  col = mix(col, vec3f(0.3, 0.9, 0.4), fill(fy, pw));
  col = mix(col, vec3f(0.9), fill(length(q) - 0.025 * L, pw));

  // A big asymmetric letter
  let fs = 0.55 * L;
  let dF = glyphF((q - vec2f(-0.5, 0.45) * L) / fs) * fs;
  col = mix(col, vec3f(1.0, 0.72, 0.25), fill(dF, pw));

  // Three dots of decreasing size, counter-clockwise
  let dc = vec2f(0.5, -0.45) * L;
  let r = 0.16 * L;
  var dd = length(q - dc - r * vec2f(0.0, 1.0)) - 0.07 * L;
  dd = min(dd, length(q - dc - r * vec2f(-0.866, -0.5)) - 0.05 * L);
  dd = min(dd, length(q - dc - r * vec2f(0.866, -0.5)) - 0.03 * L);
  col = mix(col, vec3f(0.55, 0.75, 1.0), fill(dd, pw));
  return col;
}

fn pattern(q: vec2f, sheet: f32, pw: f32) -> vec3f {
  var col: vec3f;
  switch u.space {
    case SPHERE, RP2: { col = spherePattern(q, sheet, pw); }
    case BOLZA: { col = bolzaPattern(q, pw); }
    default: { col = squarePattern(q, pw); }
  }
  if (sheet < 0.5) {
    col = decorations(col, q, pw);
  }
  return col;
}

// ---------------------------------------------------------------------------
// Ship, in its local frame (units of shipSize, nose along +y).
// The two halves have different colours so a mirrored ship is visible.

fn shipShape(p: vec2f, pw: f32) -> vec4f {
  var res = vec4f(0.0);

  let thrust = max(u.speed, 0.0);
  if (thrust > 0.0) {
    let flicker = 0.15 * sin(u.time * 47.0) + 0.1 * sin(u.time * 31.0);
    let len = (0.5 + flicker) * thrust + 0.2;
    let df = sdTriangle(p, vec2f(-0.28, -0.7), vec2f(0.0, -0.75 - len), vec2f(0.28, -0.7));
    let fc = mix(vec3f(1.0, 0.35, 0.05), vec3f(1.0, 0.9, 0.4), clamp(-df * 6.0, 0.0, 1.0));
    res = vec4f(fc, fill(df, pw));
  }

  var hull = sdTriangle(p, vec2f(0.0, 1.0), vec2f(-0.75, -0.85), vec2f(0.75, -0.85));
  hull = max(hull, -(length(p - vec2f(0.0, -1.15)) - 0.5));
  let side = select(vec3f(0.2, 0.8, 0.95), vec3f(1.0, 0.5, 0.15), p.x < 0.0);
  var hc = mix(side, vec3f(0.05), smoothstep(-0.12, -0.04, hull));
  hc = mix(hc, vec3f(0.95), fill(length(p - vec2f(0.0, 0.2)) - 0.15, pw));
  let a = fill(hull, pw);
  res = vec4f(mix(res.rgb, hc, a), max(res.a, a));
  return res;
}

// ---------------------------------------------------------------------------

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let P = params();
  let halfRes = 0.5 * min(u.resolution.x, u.resolution.y);
  var s = (pos.xy - 0.5 * u.resolution) / halfRes;
  s.y = -s.y;

  var q: vec2f;
  var sheet: f32;
  var pw: f32;
  var inside = true;

  if (u.camera == 0u) {
    // Fixed camera: the screen shows the chart itself.
    var w = s * P.fixedScale * u.zoom;
    var sh = 0.0;
    if (u.space == SPHERE) {
      // North and south hemispheres side by side.
      sh = select(0.0, 1.0, w.x > 0.0);
      w.x -= select(-1.1, 1.1, w.x > 0.0);
    }
    if (u.space == BOLZA && dot(w, w) >= 0.999) {
      return vec4f(0.0, 0.0, 0.0, 1.0);
    }
    inside = inDomain(w);
    let c = reduce(w, sh);
    q = c.x;
    sheet = c.sheet;
    pw = P.fixedScale * u.zoom / halfRes;
  } else {
    // Ship camera: follow the geodesic from the ship in this direction.
    let g = exponential(s * P.shipView * u.zoom);
    q = g.x;
    sheet = g.sheet;
    pw = P.shipView * u.zoom / halfRes / pow(max(metricDet(q), 1e-12), 0.25);
  }

  var col = pattern(q, sheet, pw);

  if (u.showCurvature != 0u) {
    let K = curvature(q);
    let tint = select(vec3f(0.2, 0.45, 1.0), vec3f(1.0, 0.45, 0.2), K > 0.0);
    col = mix(col, tint, 0.3 * tanh(min(abs(K), 8.0)));
  }

  // Balls: intrinsic distance √(Δᵀ G Δ), using the metric at the ball.
  let ballR = 0.3 * P.shipSize;
  for (var i = 0; i < BALL_COUNT; i++) {
    let b = worldRO.balls[i];
    if (b.age >= BALL_LIFE) {
      continue;
    }
    let G = mat2x2f(b.g.x, b.g.y, b.g.y, b.g.z);
    var bpw = P.shipView * u.zoom / halfRes;
    if (u.camera == 0u) {
      bpw = pw * pow(b.g.x * b.g.z - b.g.y * b.g.y, 0.25);
    }
    let fade = 1.0 - smoothstep(BALL_LIFE - 0.6, BALL_LIFE, b.age);
    for (var k = 0; k < liftCount(); k++) {
      let l = lift(k, q, sheet);
      if (abs(l.z - b.sheet) > 0.5) {
        continue;
      }
      let delta = l.xy - b.p;
      let d = sqrt(max(dot(delta, G * delta), 0.0));
      if (d > 3.0 * ballR) {
        continue;
      }
      let glow = exp(-2.0 * max(d - ballR, 0.0) / ballR) * 0.5;
      col += vec3f(1.0, 0.75, 0.3) * glow * fade;
      let core = mix(vec3f(1.0, 0.85, 0.4), vec3f(1.0, 1.0, 0.9), smoothstep(ballR, 0.0, d));
      col = mix(col, core, fill(d - ballR, bpw) * fade);
    }
  }

  // The loop: distance to its segments, over the copies of q.
  let loopMode = loopRO.mode;
  if (loopMode != LOOP_IDLE && loopRO.count > 0u) {
    var dmin = 1e30;
    let margin = vec2f(4.0 * pw);
    let n = loopRO.count;
    let blocks = (n + u32(LOOP_BLOCK) - 1u) / u32(LOOP_BLOCK);
    for (var k = 0; k < liftCount(); k++) {
      let l = lift(k, q, sheet);
      for (var bl = 0u; bl < blocks; bl++) {
        let box = loopRO.blocks[bl];
        if (any(l.xy < box.xy - margin) || any(l.xy > box.zw + margin)) {
          continue;
        }
        let end = min(n, (bl + 1u) * u32(LOOP_BLOCK));
        for (var i = bl * u32(LOOP_BLOCK); i < end; i++) {
          let p = loopRO.points[i];
          if (abs(p.sheet - l.z) < 0.5) {
            dmin = min(dmin, sdSegment(l.xy, p.x, p.next));
          }
        }
      }
    }
    let dpx = dmin / pw;
    var lc = vec3f(0.95);
    var width = 0.9;
    var alpha = 0.9;
    if (loopMode == LOOP_CONTRACTING) {
      lc = vec3f(1.0, 0.45, 0.85);
    } else if (loopMode == LOOP_CONTRACTED) {
      lc = vec3f(1.0, 0.45, 0.85);
      alpha = 0.9 * (1.0 - smoothstep(0.0, 1.5, loopRO.age));
    } else if (loopMode == LOOP_STUCK) {
      lc = vec3f(0.4, 1.0, 0.55);
      width = 1.6;
      col += lc * 0.35 * exp(-dpx / 4.0);
    }
    col = mix(col, lc, alpha * fill(dpx - width, 0.75));
  }

  // Ship: approximate local coordinates by inverting the frame at the
  // ship, which is accurate at the ship's scale.
  let ship = worldRO.ship;
  let det = ship.e1.x * ship.e2.y - ship.e2.x * ship.e1.y;
  let Einv = mat2x2f(ship.e2.y, -ship.e1.y, -ship.e2.x, ship.e1.x) * (1.0 / det);
  var spw = P.shipView * u.zoom / halfRes / P.shipSize;
  if (u.camera == 0u) {
    spw = pw * pow(metricDet(ship.p), 0.25) / P.shipSize;
  }
  for (var k = 0; k < liftCount(); k++) {
    let l = lift(k, q, sheet);
    if (abs(l.z - ship.sheet) > 0.5) {
      continue;
    }
    let lp = Einv * (l.xy - ship.p) / P.shipSize;
    if (dot(lp, lp) > 9.0) {
      continue;
    }
    let sc = shipShape(lp, spw);
    col = mix(col, sc.rgb, sc.a);
  }

  if (u.camera == 0u && !inside) {
    col *= 0.4;
  }

  return vec4f(col, 1.0);
}
