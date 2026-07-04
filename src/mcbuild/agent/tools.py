"""OpenAI-style tool schemas for the agent loop.

submit_blueprint / str_replace / edit_region / inspect / finish.
"""

SUBMIT_BLUEPRINT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_blueprint",
        "description": (
            "Validate, execute, and render a COMPLETE blueprint program, built from scratch "
            "into an empty world (it replaces any previous build). Use this to start over or "
            "make structural changes to the overall shape/footprint. For small incremental "
            "tweaks to an existing build, prefer str_replace or edit_region. On error you "
            "get a line-mapped traceback to fix. On success you get build stats and, in a "
            "follow-up message, a labeled multi-view contact-sheet render to critique."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The full blueprint Python source."},
                "design_notes": {
                    "type": "string",
                    "description": "Brief notes on what this blueprint builds or what changed since last time.",
                },
            },
            "required": ["code", "design_notes"],
        },
    },
}

STR_REPLACE_TOOL = {
    "type": "function",
    "function": {
        "name": "str_replace",
        "description": (
            "Incrementally edit the CURRENT build by find-and-replacing text in its accumulated "
            "blueprint source, like a coding agent's file-editing tool. `old_str` must match "
            "EXACTLY ONE location in the current source (whitespace and all) — include enough "
            "surrounding context (a few lines) to make the match unique; if it matches zero or "
            "multiple times you get an error and nothing changes. `new_str` replaces that match. "
            "If `submit` is true (default), the FULL resulting source is re-run from scratch and "
            "rendered — so, unlike a raw code delta, variables, helper functions, and transform "
            "contexts you defined elsewhere in the source stay intact — and this uses one edit "
            "from your budget. Set `submit` to false to stage the text edit WITHOUT building or "
            "rendering it (free, no budget used) so you can batch several str_replace calls — "
            "e.g. fix three unrelated spots — before spending a render on the combined result; "
            "the staged edits apply in order and the next str_replace/submit_blueprint call with "
            "submit=true (or the plain submit_blueprint tool) builds whatever has accumulated. "
            "On a submitted error the pre-edit build is left untouched and you get a line-mapped "
            "traceback against the patched source. On success you get updated stats and a fresh "
            "contact-sheet render. Requires a prior successful submit_blueprint."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "old_str": {
                    "type": "string",
                    "description": "Exact text to find in the current blueprint source; must be unique.",
                },
                "new_str": {
                    "type": "string",
                    "description": "Text to replace old_str with.",
                },
                "submit": {
                    "type": "boolean",
                    "description": "If true (default), build and render after this edit and spend an edit. "
                    "If false, only stage the text edit (free, no render).",
                    "default": True,
                },
                "design_notes": {"type": "string", "description": "Brief notes on what this edit changes."},
            },
            "required": ["old_str", "new_str", "design_notes"],
        },
    },
}

EDIT_REGION_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_region",
        "description": (
            "Rebuild one bounding-box region while freezing the rest of the build. The given "
            "region is cleared first, then your code runs against the current (post-clear) voxel "
            "state — so you can redo just the roof or one wing without risking regressions "
            "elsewhere. Your code runs against the EXISTING (post-clear) voxel state (not a "
            "fresh world): fresh transform stack each call, no persisted variables, pre-edit "
            "build untouched on error. Requires a prior successful submit_blueprint."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 6,
                    "maxItems": 6,
                    "description": "[x1, y1, z1, x2, y2, z2] bounding box to clear and rebuild.",
                },
                "code": {"type": "string", "description": "Blueprint snippet that rebuilds the region."},
                "design_notes": {"type": "string", "description": "Brief notes on what this edit changes."},
            },
            "required": ["region", "code", "design_notes"],
        },
    },
}

INSPECT_TOOL = {
    "type": "function",
    "function": {
        "name": "inspect",
        "description": (
            "Re-render the current build without changing it, in one of two modes. FIXED-ANGLE "
            "(default): use yaw (0-3, 90-degree steps) and optionally a cutaway or slice for a "
            "quick isometric view. FREE-CAMERA: provide BOTH camera_pos and look_at ([x,y,z] in "
            "your blueprint's coordinate space) to view from an arbitrary position and angle — "
            "e.g. to inspect a facade up close, put camera_pos a few blocks out from it with "
            "look_at pointing at it. Providing camera_pos/look_at switches to free-camera mode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "yaw": {
                    "type": "integer",
                    "description": "Camera rotation in 90-degree steps (0-3). Fixed-angle mode.",
                    "default": 0,
                },
                "cutaway": {
                    "type": "string",
                    "enum": ["none", "x", "z"],
                    "description": (
                        "Slice the build at its mid-plane on this axis to see inside (keeps the "
                        "FAR half, drops the near half). The cut face only reads as a cross-"
                        "section if the camera actually faces it — pair cutaway='x' with yaw=2 "
                        "and cutaway='z' with yaw=1; other yaws just show a smaller-looking "
                        "exterior with nothing new revealed."
                    ),
                    "default": "none",
                },
                "slice_axis": {
                    "type": "string",
                    "enum": ["x", "y", "z"],
                    "description": (
                        "Cut at an arbitrary plane on this axis (world coords, keeps the far "
                        "side). 'y' reveals a storey from that height up and reads correctly at "
                        "any yaw. 'x'/'z' have the same facing requirement as cutaway — use "
                        "yaw=2 for 'x', yaw=1 for 'z' — or the cut face won't be visible. "
                        "Overrides cutaway."
                    ),
                },
                "slice_at": {
                    "type": "integer",
                    "description": "The world coordinate of the slice plane (used with slice_axis).",
                },
                "camera_pos": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "[x, y, z] free-camera position in voxel space (with look_at).",
                },
                "look_at": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "[x, y, z] point the free camera looks toward (with camera_pos).",
                },
            },
        },
    },
}

QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "query",
        "description": (
            "Get lossless TEXT ground truth about the current build (no image). Modes: "
            "'slice' returns a one-char-per-block ASCII plan of a plane (use slice_axis + "
            "slice_at; slice_axis='y' gives a floor plan of that storey) with a material "
            "legend; 'point' returns the exact block at (x, y, z); 'histogram' returns material "
            "counts, optionally within a region. Prefer this over inspect when you need to "
            "verify exact block placement or interior layout — text is more reliable than a "
            "small render."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["slice", "point", "histogram"]},
                "slice_axis": {"type": "string", "enum": ["x", "y", "z"], "description": "For mode=slice."},
                "slice_at": {"type": "integer", "description": "World coord of the slice plane, for mode=slice."},
                "x": {"type": "integer", "description": "For mode=point."},
                "y": {"type": "integer", "description": "For mode=point."},
                "z": {"type": "integer", "description": "For mode=point."},
                "region": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 6,
                    "maxItems": 6,
                    "description": "Optional [x1,y1,z1,x2,y2,z2] for mode=histogram.",
                },
            },
            "required": ["mode"],
        },
    },
}

FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": (
            "Declare the build complete and end the session. "
            "Before finishing, verify the interior with at least one query slice and one inspect cutaway."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "A short summary of the finished build."},
                "completed_interior_check": {
                    "type": "boolean",
                    "description": "True if you verified the interior with at least one query slice "
                    "and one inspect cutaway.",
                }
            },
            "required": ["summary", "completed_interior_check"],
        },
    },
}

ALL_TOOLS = [
    SUBMIT_BLUEPRINT_TOOL,
    STR_REPLACE_TOOL,
    EDIT_REGION_TOOL,
    INSPECT_TOOL,
    QUERY_TOOL,
    FINISH_TOOL,
]
