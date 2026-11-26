# Voronoi Body Generator – Software Requirement

Nov 15, 2026 · @Matic

> Copy of the requirement document pasted as the first prompt. Source of truth for the project.

## Summary and how to use this document

Build a browser-based tool that loads an STL model and turns it into a printable Voronoi body with the same outer envelope: a Voronoi-patterned skin connected to a Voronoi strut lattice inside.

Instruction to Claude Code: treat this document as the source of truth. Where something is unclear or a decision is not covered, ask before building. Keep a short DECISIONS.md log of choices you make. Follow the build plan step by step and stop after each step so it can be tested.

## Goals and assumptions

The first target is FDM 3D printing: every output must be a watertight, manifold STL that slices without errors.

**Goals**

- Load an STL, analyse its shape and show it in a 3D preview.
- Generate a new body that keeps the original outer dimensions and never extends outside the original envelope.
- Offer several surface styles and interior styles that can be combined freely, tuned with a small set of sliders.
- Export a print-ready STL.

**Assumptions**

- Units are millimetres.
- Typical parts fit a desktop FDM printer: up to about 250 × 250 × 250 mm and up to about 500,000 input triangles.
- Default nozzle 0.4 mm, so struts and ribs should not go below about 1.2 mm by default.
- The app runs locally and opens in the browser. Hosting it on the internet is not required.
- A final generation may take up to about 60 seconds; the preview should feel interactive (a few seconds).

## Functional requirements

The user picks one surface style and one interior style, then tunes them with modifiers; every combination must work.

### Input

1. Load a binary or ASCII STL by file picker or drag and drop.
2. Validate the mesh: watertight, manifold, consistent normals, positive volume, no degenerate size.
3. If the input is broken, do not repair it. Refuse it with a clear message that lists what is wrong (for example "mesh has 14 open edges"), and suggest repairing it in another tool.
4. Show input info: bounding box dimensions (mm), triangle count, volume.

### Surface styles (outer skin)

| Style | What it does |
|---|---|
| A · Perforated shell | Solid shell of set thickness with rounded Voronoi-shaped holes through it |
| B · Open rib lattice | Only the Voronoi cell edges remain, as ribs lying on the original surface |
| C · Embossed ribs | Closed shell; cell areas are recessed by a set depth so the ribs stand at the original surface (envelope unchanged) |
| None | No skin; the interior lattice is simply cut by the original envelope |
| Solid | Keep a plain solid shell of set thickness |

### Interior styles

| Style | What it does |
|---|---|
| Struts | 3D Voronoi cell edges turned into round beams |
| Solid | Interior stays filled (useful with surface style C) |
| Hollow | Interior is empty; only the skin remains |

### Modifiers

- **Grading:** cell size and/or thickness vary smoothly. Modes: off, by distance from the skin, or along the X, Y or Z axis. Applies to skin and interior.
- **Blend radius:** 0 gives sharp joints; higher values give smooth, organic, filleted joints between struts, and between struts and the skin.
- **Connection:** interior struts must merge into the skin so the part is one connected body. With style B the skin ribs act as the outermost struts.
- **Solid base:** optional flat solid layer at the bottom (lowest Z) for bed adhesion.
- **Random seed:** same settings and seed always give the identical result.

### Output

1. The output must be watertight, manifold and a single connected body. Remove floating fragments and report how many were removed.
2. If the program's own processing produces defects, repair them automatically before export. If repair fails, show an error and do not export.
3. Export binary STL. Export filename defaults to `<input name>_voronoi.stl`.
4. Save the settings used as a small JSON file next to the export, so a result can be reproduced.

## Settings

Show only the settings that apply to the selected styles; all lengths in mm. The defaults below are starting points for FDM with a 0.4 mm nozzle.

| Setting | Applies to | Range | Default |
|---|---|---|---|
| Surface style | all | A / B / C / None / Solid | A |
| Interior style | all | Struts / Solid / Hollow | Struts |
| Shell thickness (mm) | A, C, Solid | 0.8 – 10 | 2.0 |
| Surface cell size (mm) | A, B, C | 3 – 50 | 10 |
| Hole size (% of cell) | A | 10 – 90 | 60 |
| Rib width (mm) | B, C | 0.8 – 10 | 2.0 |
| Emboss depth (mm) | C | 0.4 – shell thickness | 0.8 |
| Interior cell size (mm) | Struts | 3 – 60 | 12 |
| Strut diameter (mm) | Struts | 0.8 – 10 | 2.0 |
| Blend radius (mm) | Struts, B | 0 – 5 | 0.5 |
| Grading mode | all patterned | Off / From skin / X / Y / Z | Off |
| Grading strength (%) | when grading on | 0 – 100 | 50 |
| Grading target | when grading on | Cell size / Thickness / Both | Cell size |
| Grading direction | when grading on | Denser at start / Denser at end | Denser at start (near skin) |
| Solid base thickness (mm) | all | 0 (off) – 5 | 0 |
| Random seed | all | integer + Randomize button | 1 |
| Nozzle diameter (mm) | printability warnings | 0.2 – 1.0 | 0.4 |
| Final quality | export | Draft / Normal / Fine | Normal |

**Printability warnings:** show a non-blocking warning when a strut, rib or wall is thinner than 3 × nozzle diameter, or a hole is smaller than 5 × nozzle diameter.

**Quality:** the detail level (voxel size) is chosen automatically from the part size and the thinnest feature (target: at least 4 voxels across the thinnest strut), capped by a memory budget. If the cap forces a coarser result, warn the user.

## Geometry approach

Build all geometry as signed distance fields (SDF) on a voxel grid and extract the final mesh with marching cubes. Do not try to join tube meshes directly: that approach breaks watertightness.

1. **Body field:** compute the signed distance field of the input mesh on a padded grid (negative inside).
2. **Seed points:** place interior seeds inside the body with Poisson-disk sampling, spacing = interior cell size (varied by grading). Place surface seeds on the mesh surface with spacing = surface cell size. Use the random seed for every random step.
3. **Interior struts:** compute the 3D Voronoi diagram of the interior seeds (for example via Delaunay). Take its finite edges, clip them to the body, and evaluate a capsule distance per edge (radius = strut diameter / 2, varied by grading). Combine edges with smooth-min using the blend radius.
4. **Surface pattern:** in a band of shell thickness inside the surface, evaluate the Worley field F2 − F1 (distance to second-nearest minus nearest surface seed). Small F2 − F1 marks the Voronoi edges on the surface.
   - A: shell minus the cell interiors where F2 − F1 exceeds a threshold set by hole size, with rounded hole edges.
   - B: keep only where F2 − F1 is below the rib width, giving ribs.
   - C: full shell, with cell interiors recessed by the emboss depth from the outside.
5. **Combine:** union of skin and interior (smooth union with blend radius), plus the optional solid base, then intersect with the body field so nothing goes outside the original envelope.
6. **Mesh:** run marching cubes, keep the largest connected body, remove fragments, validate, auto-repair small defects, and optionally decimate to keep the STL size reasonable.

Two resolutions use the same pipeline: a coarse preview grid (target under 5 seconds) and a fine final grid (target under 60 seconds). Run heavy work off the UI thread, with a progress bar and a Cancel button. Evaluate fields in chunks, and only near geometry where possible, to stay inside the memory budget.

## User interface

One screen: a settings panel on the left, a large 3D viewport in the centre, and a toolbar on top.

**Toolbar:** Load STL · Generate final · Export STL · Reset settings.

**Settings panel:** grouped as Surface, Interior, Modifiers and Print, with the settings from the table above. Sliders show a numeric value that can also be typed. Settings that don't apply to the chosen style are hidden.

**Viewport (three.js or similar)**

- Orbit, pan and zoom with the mouse; buttons for front, top, side and isometric views; fit to view.
- View mode: Original, Result, or Overlay (original as a transparent ghost over the result).
- Clipping plane along X, Y or Z with a slider, to look inside the part.
- Optional overhang highlight: colour faces that overhang more than 45° relative to the build direction (Z up).
- Bounding box with dimensions in mm.

**Preview behaviour:** after a slider is released (debounced about 300 ms), regenerate the coarse preview automatically. "Generate final" builds the fine version with a progress bar and Cancel. Export is enabled only after a final result that passed validation.

**Info bar:** input and output dimensions, triangle counts, generation time, warnings (printability, quality capped, fragments removed).

**Errors:** shown in plain language in a dismissable panel, never only in the console.

## Tech stack and architecture

Recommended: a local web app, with a Python geometry engine behind a browser front end, started with one command. The language is open, so Claude Code may propose an alternative, but it must explain the trade-off first.

| Part | Recommendation | Why |
|---|---|---|
| Front end | TypeScript + three.js, built with Vite | Mature 3D viewer, clipping planes, STL loader |
| Geometry engine | Python: numpy, scipy (Voronoi/Delaunay, KD-tree), scikit-image (marching cubes), trimesh (STL I/O, validation, repair) | Proven libraries for every step of the pipeline |
| Connection | FastAPI on localhost; jobs run in a background worker with progress and cancel | Keeps the UI responsive during long jobs |
| Start | One command (for example ./start.sh / start.bat) that installs dependencies and opens the browser | Easy to run on Windows, macOS and Linux |

Alternative to consider: a fully in-browser version (geometry compiled to WebAssembly, for example with the manifold-3d library) can be hosted as a static website, but it is harder to build and slower for large parts.

**Code requirements:** geometry engine separate from UI code; settings as one typed schema shared by UI, API and the saved JSON; unit tests for the geometry engine (pytest); a README with setup and usage.

## Out of scope

Do not build these in this version:

- Closed-cell (foam-like) Voronoi walls.
- Repair of broken input STL files.
- Stress analysis (FEA) or load-based optimisation.
- Resin and SLS specifics such as drain holes; multi-material output.
- User accounts, cloud hosting, mobile layout.
- Slicing or printer control.

## Acceptance criteria

The first version is done when all of these pass, checked by automated tests where possible:

- [ ] A valid STL up to 500,000 triangles loads and displays in under 5 seconds.
- [ ] A broken STL (open edges, non-manifold) is refused with a message naming the problem.
- [ ] Every combination of surface style × interior style generates without errors on the test models.
- [ ] The output bounding box matches the input within ± one voxel (at most 0.3 mm at Normal quality) on every axis, and no geometry lies outside the input envelope.
- [ ] The output passes trimesh checks: watertight, winding consistent, positive volume, one connected body.
- [ ] Final generation of a 100 mm part at default settings takes at most 60 seconds on a typical laptop; the preview takes at most 5 seconds.
- [ ] The same settings and seed give a byte-identical STL.
- [ ] The exported STL slices without errors in PrusaSlicer or Cura.
- [ ] Test models generated in code are included: 50 mm cube, 60 mm sphere, a 100 mm bracket with a thin flange.

## Build plan

Build in these steps, in order. After each step, stop, explain how to test it, and wait for confirmation.

1. **Scaffold:** project structure, start script, empty UI with viewport, test models generated in code.
2. **Load and validate:** STL loading, validation with clear errors, display, info bar, view controls, clipping plane.
3. **SDF core:** body distance field, marching cubes, round trip (input → field → mesh) matches the input within one voxel. Add surface styles Solid and None, and interior styles Solid and Hollow.
4. **Interior struts:** seeds, 3D Voronoi edges, capsules, blend radius.
5. **Surface styles A, B, C:** Worley-based patterns and their connection to the interior.
6. **Modifiers:** grading, solid base, seed, printability warnings.
7. **Preview and final pipeline:** background jobs, progress, cancel, auto quality and memory cap.
8. **Export and validation:** auto-repair of own defects, STL plus settings JSON export, all acceptance tests passing.

**Phase 2 (later):** saved presets, weight and material-savings stats (PLA at 1.24 g/cm³), solid keep-out zones around holes or flat faces, overhang-aware strut orientation.
