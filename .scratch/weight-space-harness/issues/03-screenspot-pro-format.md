# ScreenSpot Pro format and the Qwen3.5 grounding prompt

Type: research
Status: resolved

## Question

What is the data layout and the scoring rule of ScreenSpot Pro, and how does Qwen3.5 expect a
grounding prompt and write a coordinate?

## Answer

Researched on 2026-11-10. Sources: the dataset `likaixin/ScreenSpot-Pro` on Hugging Face and the
repository `likaixin2000/ScreenSpot-Pro-GUI-Grounding` on GitHub. Both have an MIT licence.

**Dataset.**

- 3.38 GB. `annotations/` holds 26 JSON files, one per application, named `<app>_<platform>.json`.
  `images/` holds 26 folders of PNG files. `img_filename` is relative to `images/`.
- A record has ten fields: `img_filename`, `bbox`, `instruction`, `instruction_cn`, `id`,
  `application`, `platform`, `img_size`, `ui_type`, `group`.
- `bbox` is `[x1, y1, x2, y2]` in pixels of the original image. `img_size` is `[width, height]`.
- `ui_type` is `icon` or `text`.
- 1581 samples by the README table: 604 icon and 977 text. `eval.yaml` says 1,585. Count the records
  after the download.

| Group | Samples | Applications |
|---|---|---|
| Dev | 299 | VSC 55, PyCharm 78, Android Studio 80, Quartus 45, VMware 41 |
| Creative | 341 | Photoshop 51, Premiere 52, Illustrator 31, Blender 71, FL Studio 57, Unreal 35, DaVinci 44 |
| CAD | 261 | AutoCAD 34, SolidWorks 77, Inventor 70, Vivado 80 |
| Scientific | 254 | MATLAB 93, Origin 62, Stata 49, EViews 50 |
| Office | 230 | Word 84, PowerPoint 82, Excel 64 |
| OS | 196 | Windows 81, macOS 65, Linux 50 |

**Scoring.** A prediction is correct when the click point is inside the bbox, edges included. A reply
with no point is wrong. The score is a micro average over all samples.

**Model format.** The repository has a Qwen3.5 wrapper at `models/qwen3_5.py` with default path
`Qwen/Qwen3.5-4B`. Read the file for the exact prompt string.

- The system message defines one `computer_use` tool inside `<tools></tools>`. The tool allows only
  `left_click` with a `coordinate` and states a 1000x1000 screen.
- The user turn is the image, then the instruction text.
- The reply is JSON inside `<tool_call>` tags. The parser takes `arguments.coordinate` and divides
  by 1000. A four-value box becomes its centre.
- Resolution on the Hugging Face path: `max_pixels` default 8,294,400 (env `QWEN3_5_MAX_PIXELS`).
- Generation on the Hugging Face path: greedy, `max_new_tokens=4096`.
- The wrapper does not set `enable_thinking`, so thinking is probably on. No source states the
  setting behind the published 60.3.

**Open point for the baseline ticket.** Run with thinking on and with thinking off. The setting that
gives a score near 60.3 is the reference setting.
