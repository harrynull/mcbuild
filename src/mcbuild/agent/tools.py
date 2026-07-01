"""OpenAI-style tool schemas for the agent loop: submit_blueprint / inspect / finish."""

SUBMIT_BLUEPRINT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_blueprint",
        "description": (
            "Validate, execute, and render a complete blueprint program. The blueprint "
            "always builds the whole structure from scratch (it replaces any previous "
            "build, it does not patch it incrementally). On error you get back a "
            "line-mapped traceback to fix. On success you get build stats and, in a "
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

INSPECT_TOOL = {
    "type": "function",
    "function": {
        "name": "inspect",
        "description": (
            "Re-render the current build from a specific angle or as a cutaway, without "
            "changing it. Use this to closely inspect a detail before deciding on the next "
            "blueprint revision."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "yaw": {
                    "type": "integer",
                    "description": "Camera rotation in 90-degree steps (0-3).",
                    "default": 0,
                },
                "cutaway": {
                    "type": "string",
                    "enum": ["none", "x", "z"],
                    "description": "Slice the build at its mid-plane on this axis to see inside.",
                    "default": "none",
                },
            },
        },
    },
}

FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Declare the build complete and end the session.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "A short summary of the finished build."},
            },
            "required": ["summary"],
        },
    },
}

ALL_TOOLS = [SUBMIT_BLUEPRINT_TOOL, INSPECT_TOOL, FINISH_TOOL]
