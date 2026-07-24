import { useEffect, useRef } from "react";

/**
 * Fixed, full-viewport WebGL fluid simulation background.
 * Reacts to the system cursor (and touch) exactly like PavelDoGreat's
 * WebGL-Fluid-Simulation. The actual simulation code lives untouched at
 * /public/vendor/fluid-sim.js (only the pointer-tracking was adapted so it
 * keeps working while the canvas sits *behind* the dashboard content with
 * pointer-events: none — see comments in that file).
 *
 * Drop-in replacement for the old OceanBackground: render it once, high in
 * the layout, e.g. in App.tsx:
 *   <FluidBackground />
 */
export default function FluidBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    function loadSim() {
      // Guard against double-mount (React 18 StrictMode in dev) so we never
      // attach the simulation's global listeners/WebGL context twice.
      const w = window as any;
      if (w.__fluidSimLoaded) return;
      w.__fluidSimLoaded = true;

      const script = document.createElement("script");
      // Cache-bust on reload after a context loss, so the browser actually
      // re-executes the script instead of serving it from cache/no-op.
      script.src = "/vendor/fluid-sim.js?t=" + Date.now();
      script.async = true;
      document.body.appendChild(script);
    }

    // The page mounts several other WebGL canvases (the interactive 3D plot
    // cards). Browsers cap how many WebGL contexts can exist at once — once
    // that cap is hit, the *oldest* context (this background, since it's
    // mounted first at the App root) can get silently evicted by the
    // browser, leaving a permanently blank canvas. Listen for that and
    // reinitialize instead of just staying blank.
    function handleContextLost(e: Event) {
      e.preventDefault();
      const w = window as any;
      w.__fluidSimLoaded = false;
      // Give the browser a tick to finish tearing down the old context
      // before we ask for a new one.
      setTimeout(loadSim, 50);
    }

    canvas.addEventListener("webglcontextlost", handleContextLost, false);
    loadSim();

    return () => {
      canvas.removeEventListener("webglcontextlost", handleContextLost);
      // The simulation is meant to live for the lifetime of the page (same
      // as the old background), so we intentionally don't tear it down on
      // a normal unmount — App only mounts this once at the root.
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "fixed",
        inset: 0,
        width: "100vw",
        height: "100vh",
        zIndex: 0,
        display: "block",
        pointerEvents: "none",
        background: "#050403",
      }}
    />
  );
}
