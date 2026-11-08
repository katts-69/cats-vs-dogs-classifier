# Design

Both versions of the app follow this file: the online page ([web/](web/), plain HTML, CSS and
JavaScript) and the local Gradio app ([app.py](app.py)). The app looks like a retro 1990s desktop: each part of the page is a square,
bevelled window with a striped title bar, on a dotted teal wallpaper. The goal is a demo that is
fun and memorable, while the result stays readable at a glance.

## Colours

Use only these. The same colours are used in light and dark mode.

| Name | Value | Used for |
|---|---|---|
| Ink | `#1b1230` | text, borders, hard shadows |
| Cream / Cream 2 | `#f4ecd0` / `#e2d6a8` | window faces / labels, secondary buttons |
| Light / Shade | `#fffbe8` / `#b8aa7a` | bevel highlight (top-left) / bevel shade (bottom-right) |
| Teal / Dark teal | `#2a9d9d` / `#0e4f55` | wallpaper dots / page background |
| Magenta | `#e0338f` | title bars, the primary button, confidence bars |
| Yellow | `#ffd23f` | the alert window's title bar, terminal prompts |
| Terminal | `#0b2b30` with `#9ff3d0` text | the model-specs readout |

## Type

- **Pixel font** ("Press Start 2P"): title bars, the page heading, buttons, the predicted label.
- **Monospace** (ui-monospace, Consolas): all other text, 13px with 1.45 line height.

## Windows

- Square corners everywhere; no rounded corners, gradients or blur. The only exceptions are the
  hard-edged stripes on title bars and the wallpaper dots.
- A window has a 2px ink border, a 2px light/shade bevel inside, and a 4px hard shadow outside.
- Title bars are 28px tall, magenta stripes (yellow for alerts), with a file-style name such as
  `INPUT.BMP` and decorative `_` and `x` buttons.
- 14px padding inside windows. The page is at most 1120px wide and centred.

## Components

- **Buttons:** pixel font, 2px ink border, 3px hard shadow; they press down (shadow disappears) when
  clicked. The main action is magenta, the others cream.
- **Result:** the predicted label in the pixel font, then one 14px magenta bar per class with an ink
  border.
- **Focus:** no blue glow; keyboard focus shows a dashed ink outline.

## Layout

| Window | Content |
|---|---|
| `CAT_OR_DOG.EXE` | heading and a one-paragraph description |
| `INPUT.BMP` | photo upload, Clear and Classify buttons |
| `RESULT.TXT` | prediction, plus a terminal readout of the model's specs |
| `SYSTEM ALERT` | the model's limitations |

Input and result sit side by side on wide screens and stack on phones.

## Behaviour

- Classify as soon as a photo is chosen or dropped; no extra click needed.
- Keep the wording plain. The limitations note must stay visible: it's part of being honest about
  the model.
