# Fluid background + Nav fix

## What was actually wrong

**1. Fluid background only showed color where the cursor had literally been**

`fluid-sim.js` (from your Fluids_v3 wallpaper) has an ambient "auto-splat"
feature that's meant to keep the background alive on its own — but it's
driven by a Lively Wallpaper settings-panel variable, `_randomSplats`, that
defaulted to `false` and was only ever switched on by Lively's UI. On a plain
webpage nothing ever turns it on, so the canvas stayed black except for the
trail your cursor happened to leave — which is exactly the patchy blob in
your screenshot, not a rendering bug.

Fix: `_randomSplats` now defaults to `true`, and the auto-splat volume is
turned down (1–3 splats every ~4s instead of 5–24 every 3.5s, which was
tuned for a fullscreen wallpaper app, not a page background) so the whole
canvas gets gentle ambient motion everywhere, not just under the cursor.

**2. Navbar was almost invisible against the fluid background**

The nav was reusing `.liquid-glass`, which is `rgba(255,255,255,0.02)` —
built for content cards that sit over an already-darkened part of the page.
Floating directly over the busiest, most colorful part of the fluid sim, that
same near-transparent glass just let the background bleed through, so the
bar never read as a clean pill.

Fix: added a dedicated `.nav-glass` class (real dark backing + stronger
blur, `liquid-glass.css`) and pointed `Nav.tsx` at it instead — same floating
pill/logo/links/mobile-drawer structure you already had, just legible now no
matter what's moving behind it.

## Files in this package

```
public/vendor/fluid-sim.js          fixed — ambient motion enabled + tuned
src/liquid-glass.css                edited — added .nav-glass
src/components/Nav.tsx              edited — uses .nav-glass instead of .liquid-glass
src/components/FluidBackground.tsx  unchanged — included again just in case
```

## Apply

Copy these over the same paths in your project (overwrite), then:

```bash
npm run dev
```

No new dependencies, no other files touched.
