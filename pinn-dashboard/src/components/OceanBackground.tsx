import { useEffect, useRef } from "react";
import * as THREE from "three";

/**
 * Fixed, full-viewport animated Red Sea underwater background.
 * Drop this file into: pinn-dashboard/src/components/OceanBackground.tsx
 * Then render it once, high in your layout, e.g. in App.tsx:
 *   <OceanBackground />
 *   <Navbar />
 *   <main>...</main>
 */
export default function OceanBackground() {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    // IMPORTANT: use window dimensions, not mount.clientWidth/Height —
    // a `position: fixed` div has not been laid out yet on the same tick
    // this effect runs, so clientWidth/Height would read 0.
    let width = window.innerWidth;
    let height = window.innerHeight;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x0a0403, 0.028);

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    camera.position.set(0, 2, 15);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";
    mount.appendChild(renderer.domElement);

    // Lighting — warm red-orange, evoking sunlight through Red Sea water
    scene.add(new THREE.AmbientLight(0x4a1f18, 1.1));
    const sunShaft = new THREE.PointLight(0xff8f6b, 2.2, 45);
    sunShaft.position.set(0, 14, 4);
    scene.add(sunShaft);
    const deepGlow = new THREE.PointLight(0xd9432c, 1.6, 40);
    deepGlow.position.set(-6, -6, -8);
    scene.add(deepGlow);

    // Glowing core
    const coreGeo = new THREE.IcosahedronGeometry(1.1, 2);
    const coreMat = new THREE.MeshStandardMaterial({
      color: 0x2a0f0c,
      emissive: 0xb8402a,
      emissiveIntensity: 0.55,
      metalness: 0.35,
      roughness: 0.4,
    });
    const core = new THREE.Mesh(coreGeo, coreMat);
    scene.add(core);
    const coreWire = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.35, 1),
      new THREE.MeshBasicMaterial({ color: 0xff9c85, wireframe: true, transparent: true, opacity: 0.22 })
    );
    scene.add(coreWire);

    // Undulating sea floor
    const floorGeo = new THREE.PlaneGeometry(50, 50, 60, 60);
    floorGeo.rotateX(-Math.PI / 2);
    const floorPos = floorGeo.attributes.position as THREE.BufferAttribute;
    const floorBase = new Float32Array(floorPos.array.length);
    floorBase.set(floorPos.array as Float32Array);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x1a0a08,
      emissive: 0x3a1108,
      emissiveIntensity: 0.25,
      metalness: 0.5,
      roughness: 0.6,
      transparent: true,
      opacity: 0.5,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.position.y = -8;
    scene.add(floor);

    // Drifting particles
    const particleCount = 450;
    const pGeo = new THREE.BufferGeometry();
    const pPos = new Float32Array(particleCount * 3);
    const pSpeed = new Float32Array(particleCount);
    for (let i = 0; i < particleCount; i++) {
      pPos[i * 3] = (Math.random() - 0.5) * 40;
      pPos[i * 3 + 1] = (Math.random() - 0.5) * 30;
      pPos[i * 3 + 2] = (Math.random() - 0.5) * 40;
      pSpeed[i] = 0.3 + Math.random() * 0.6;
    }
    pGeo.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
    const particles = new THREE.Points(
      pGeo,
      new THREE.PointsMaterial({ color: 0xffab8f, size: 0.05, transparent: true, opacity: 0.45 })
    );
    scene.add(particles);

    const clock = new THREE.Clock();
    let frameId: number;

    function animate() {
      frameId = requestAnimationFrame(animate);
      const t = clock.getElapsedTime();

      core.rotation.y = t * 0.22;
      core.rotation.x = t * 0.1;
      coreWire.rotation.y = -t * 0.13;
      const corePulse = 1 + Math.sin(t * 1.4) * 0.05;
      core.scale.set(corePulse, corePulse, corePulse);

      const fArr = (floorGeo.attributes.position as THREE.BufferAttribute).array as Float32Array;
      for (let k = 0; k < fArr.length; k += 3) {
        const fx = floorBase[k];
        const fz = floorBase[k + 2];
        fArr[k + 1] = Math.sin(fx * 0.2 + t * 0.4) * 0.5 + Math.cos(fz * 0.18 + t * 0.35) * 0.4;
      }
      (floorGeo.attributes.position as THREE.BufferAttribute).needsUpdate = true;
      floorGeo.computeVertexNormals();

      const pArr = (pGeo.attributes.position as THREE.BufferAttribute).array as Float32Array;
      for (let m = 0; m < particleCount; m++) {
        pArr[m * 3 + 1] += pSpeed[m] * 0.01;
        if (pArr[m * 3 + 1] > 15) pArr[m * 3 + 1] = -15;
        pArr[m * 3] += Math.sin(t * 0.4 + m) * 0.0025;
      }
      (pGeo.attributes.position as THREE.BufferAttribute).needsUpdate = true;

      sunShaft.intensity = 2.0 + Math.sin(t * 0.7) * 0.35;

      const autoAngle = t * 0.015;
      camera.position.x = Math.sin(autoAngle) * 15;
      camera.position.z = Math.cos(autoAngle) * 15;
      camera.position.y = 2 + Math.sin(t * 0.1) * 0.5;
      camera.lookAt(0, 0, 0);

      renderer.render(scene, camera);
    }
    animate();

    function handleResize() {
      width = window.innerWidth;
      height = window.innerHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    }
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener("resize", handleResize);
      mount.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  return (
    <div
      ref={mountRef}
      style={{
        position: "fixed",
        inset: 0,
        width: "100vw",
        height: "100vh",
        zIndex: 0,
        pointerEvents: "none",
      }}
    />
  );
}
