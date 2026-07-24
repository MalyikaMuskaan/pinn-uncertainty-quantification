"""
background.py
-------------
Returns a self-contained HTML string that renders an animated Three.js
Red Sea underwater background via st.components.v1.html().

Visual elements
---------------
- Deep red-to-navy gradient scene (ocean descent feeling)
- 6 animated caustic light rays fanning down from the surface
- 180 floating bioluminescent particles drifting upward with random drift
- A slow sinusoidal camera-bob to simulate buoyancy
- Fog/depth attenuation on particles near the bottom

Performance
-----------
- Single <canvas> via Three.js WebGLRenderer (no postprocessing)
- Geometry reuse: all particles share one BufferGeometry
- requestAnimationFrame loop, no blocking operations
- Pointer-events: none so clicks pass through to Streamlit widgets
"""


def get_background_html(height: int = 100) -> str:
    """Return the full HTML string for the Red Sea background canvas.

    Parameters
    ----------
    height : CSS vh value for the fixed overlay height (default 100)
    """
    return f"""
<div id="sea-bg" style="
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: {height}vh;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"
        crossorigin="anonymous"></script>
<script>
(function () {{
  'use strict';

  const container = document.getElementById('sea-bg');
  if (!container) return;

  // ---- Renderer --------------------------------------------------------
  const renderer = new THREE.WebGLRenderer({{ antialias: false, alpha: true }});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.setSize(container.clientWidth || window.innerWidth,
                   container.clientHeight || window.innerHeight);
  renderer.setClearColor(0x000000, 0);
  container.appendChild(renderer.domElement);

  // ---- Scene + Camera --------------------------------------------------
  const scene  = new THREE.Scene();
  scene.fog    = new THREE.FogExp2(0x1a0005, 0.035);

  const camera = new THREE.PerspectiveCamera(
    65,
    (container.clientWidth || window.innerWidth) /
    (container.clientHeight || window.innerHeight),
    0.1, 200
  );
  camera.position.set(0, 0, 18);

  // ---- Background gradient plane ---------------------------------------
  const bgGeo = new THREE.PlaneGeometry(200, 200);
  const bgMat = new THREE.MeshBasicMaterial({{
    color: 0x0d0010,
    side: THREE.FrontSide,
    depthWrite: false,
  }});
  const bgMesh = new THREE.Mesh(bgGeo, bgMat);
  bgMesh.position.z = -60;
  scene.add(bgMesh);

  // ---- Caustic light rays ----------------------------------------------
  const rayCount  = 6;
  const rayGeo    = new THREE.PlaneGeometry(0.35, 28);
  const rayMat    = new THREE.MeshBasicMaterial({{
    color:       0xff2244,
    transparent: true,
    opacity:     0.06,
    depthWrite:  false,
    blending:    THREE.AdditiveBlending,
    side:        THREE.DoubleSide,
  }});
  const rays = [];
  for (let i = 0; i < rayCount; i++) {{
    const mesh = new THREE.Mesh(rayGeo, rayMat.clone());
    const angle = (i / rayCount) * Math.PI * 0.6 - Math.PI * 0.15;
    mesh.position.set(Math.sin(angle) * 6, 4, -10 - i * 2);
    mesh.rotation.z = angle * 0.9;
    mesh.userData.phase = i * 1.1;
    rays.push(mesh);
    scene.add(mesh);
  }}

  // ---- Floating particles ----------------------------------------------
  const N       = 180;
  const pos     = new Float32Array(N * 3);
  const vel     = new Float32Array(N * 3);   // drift per frame
  const phases  = new Float32Array(N);

  function randRange(a, b) {{ return a + Math.random() * (b - a); }}

  for (let i = 0; i < N; i++) {{
    pos[i*3  ] = randRange(-22, 22);
    pos[i*3+1] = randRange(-14, 14);
    pos[i*3+2] = randRange(-30, 0);
    vel[i*3  ] = randRange(-0.003, 0.003);   // slow x drift
    vel[i*3+1] = randRange( 0.004, 0.010);   // upward drift
    vel[i*3+2] = 0;
    phases[i]  = Math.random() * Math.PI * 2;
  }}

  const ptGeo = new THREE.BufferGeometry();
  ptGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));

  const ptMat = new THREE.PointsMaterial({{
    color:       0xff5577,
    size:        0.18,
    transparent: true,
    opacity:     0.55,
    depthWrite:  false,
    blending:    THREE.AdditiveBlending,
    sizeAttenuation: true,
  }});
  const points = new THREE.Points(ptGeo, ptMat);
  scene.add(points);

  // ---- Animation loop --------------------------------------------------
  let t = 0;
  function animate() {{
    requestAnimationFrame(animate);
    t += 0.012;

    // Camera bob
    camera.position.y = Math.sin(t * 0.4) * 0.3;
    camera.position.x = Math.sin(t * 0.27) * 0.15;

    // Caustic rays pulse
    rays.forEach((ray, idx) => {{
      const ph = ray.userData.phase;
      ray.material.opacity = 0.04 + 0.05 * Math.abs(Math.sin(t * 0.7 + ph));
      ray.rotation.z += 0.0003 * Math.sin(t * 0.5 + ph);
    }});

    // Drift particles
    for (let i = 0; i < N; i++) {{
      pos[i*3  ] += vel[i*3  ] + 0.001 * Math.sin(t + phases[i]);
      pos[i*3+1] += vel[i*3+1];
      pos[i*3+2] += 0.0005 * Math.cos(t * 0.8 + phases[i]);

      // Wrap vertically
      if (pos[i*3+1] > 15) {{
        pos[i*3+1] = -15;
        pos[i*3  ] = randRange(-22, 22);
      }}
      // Wrap horizontally
      if (pos[i*3] >  23) pos[i*3] = -23;
      if (pos[i*3] < -23) pos[i*3] =  23;
    }}
    ptGeo.attributes.position.needsUpdate = true;

    renderer.render(scene, camera);
  }}
  animate();

  // ---- Resize handler --------------------------------------------------
  window.addEventListener('resize', () => {{
    const w = container.clientWidth  || window.innerWidth;
    const h = container.clientHeight || window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }});
}})();
</script>
"""
