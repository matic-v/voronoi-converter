// AUTO-GENERATED from backend/voronoi_engine/settings.py - do not edit.
// Regenerate with: python -m voronoi_engine.export_schema

export const SETTINGS_VERSION = 1;

export type SurfaceStyle = "perforated" | "rib_lattice" | "embossed" | "none" | "solid";
export type InteriorStyle = "struts" | "solid" | "hollow";
export type GradingMode = "off" | "from_skin" | "x" | "y" | "z";
export type GradingTarget = "cell_size" | "thickness" | "both";
export type GradingDirection = "denser_at_start" | "denser_at_end";
export type Quality = "draft" | "normal" | "fine";
export type Smoothness = "off" | "smooth" | "extra";
export type Cleanup = "off" | "light" | "medium" | "strong";
export type ExportFormat = "3mf" | "stl";

export interface Settings {
  surface_style: SurfaceStyle;
  shell_thickness: number;
  surface_cell_size: number;
  hole_size: number;
  rib_width: number;
  emboss_depth: number;
  edge_rounding: number;
  interior_style: InteriorStyle;
  interior_cell_size: number;
  strut_diameter: number;
  blend_radius: number;
  grading_mode: GradingMode;
  grading_strength: number;
  grading_target: GradingTarget;
  grading_direction: GradingDirection;
  solid_base_thickness: number;
  random_seed: number;
  keep_edges: number[];
  keep_width: number;
  bed_normal: number[];
  self_supporting: boolean;
  cell_stretch: number;
  overhang_angle: number;
  max_bridge: number;
  pointed_holes: boolean;
  nozzle_diameter: number;
  quality: Quality;
  smoothness: Smoothness;
  cleanup: Cleanup;
  export_format: ExportFormat;
}

export type SettingKey = keyof Settings;

export interface SettingField {
  key: SettingKey;
  kind: "enum" | "float" | "int" | "bool" | "list";
  group: "surface" | "interior" | "modifiers" | "keep" | "printing" | "print";
  label: string;
  unit: string;
  default: string | number | boolean | number[];
  /** Set by clicking in the viewport, not by a slider. */
  picked_in_viewport?: boolean;
  /** Custom control for settings picked in the viewport. */
  widget?: "kept_edges" | "bed_face";
  min?: number;
  max?: number;
  step?: number;
  /** Upper limit is additionally capped by the current value of this field. */
  max_field?: SettingKey;
  options?: { value: string; label: string }[];
  /** Visible when ANY condition holds; a condition holds when ALL entries match. */
  show_if?: Partial<Record<SettingKey, string[]>>[];
}

export const DEFAULT_SETTINGS: Settings = {
  "surface_style": "perforated",
  "shell_thickness": 2.0,
  "surface_cell_size": 10.0,
  "hole_size": 60.0,
  "rib_width": 2.0,
  "emboss_depth": 0.8,
  "edge_rounding": 0.5,
  "interior_style": "struts",
  "interior_cell_size": 12.0,
  "strut_diameter": 2.0,
  "blend_radius": 0.5,
  "grading_mode": "off",
  "grading_strength": 50.0,
  "grading_target": "cell_size",
  "grading_direction": "denser_at_start",
  "solid_base_thickness": 0.0,
  "random_seed": 1,
  "keep_edges": [],
  "keep_width": 2.0,
  "bed_normal": [
    0.0,
    0.0,
    -1.0
  ],
  "self_supporting": true,
  "cell_stretch": 2.0,
  "overhang_angle": 45.0,
  "max_bridge": 8.0,
  "pointed_holes": true,
  "nozzle_diameter": 0.4,
  "quality": "normal",
  "smoothness": "off",
  "cleanup": "medium",
  "export_format": "3mf"
};

export const SETTING_FIELDS: SettingField[] = [
  {
    "key": "surface_style",
    "group": "surface",
    "label": "Surface style",
    "unit": "",
    "default": "perforated",
    "kind": "enum",
    "options": [
      {
        "value": "perforated",
        "label": "A · Perforated shell"
      },
      {
        "value": "rib_lattice",
        "label": "B · Open rib lattice"
      },
      {
        "value": "embossed",
        "label": "C · Embossed ribs"
      },
      {
        "value": "none",
        "label": "None"
      },
      {
        "value": "solid",
        "label": "Solid"
      }
    ]
  },
  {
    "key": "shell_thickness",
    "group": "surface",
    "label": "Shell thickness",
    "unit": "mm",
    "step": 0.1,
    "show_if": [
      {
        "surface_style": [
          "perforated",
          "embossed",
          "solid"
        ]
      }
    ],
    "default": 2.0,
    "kind": "float",
    "min": 0.8,
    "max": 10
  },
  {
    "key": "surface_cell_size",
    "group": "surface",
    "label": "Surface cell size",
    "unit": "mm",
    "step": 0.5,
    "show_if": [
      {
        "surface_style": [
          "perforated",
          "rib_lattice",
          "embossed"
        ]
      }
    ],
    "default": 10.0,
    "kind": "float",
    "min": 3,
    "max": 50
  },
  {
    "key": "hole_size",
    "group": "surface",
    "label": "Hole size (of cell)",
    "unit": "%",
    "step": 1,
    "show_if": [
      {
        "surface_style": [
          "perforated"
        ]
      }
    ],
    "default": 60.0,
    "kind": "float",
    "min": 10,
    "max": 90
  },
  {
    "key": "rib_width",
    "group": "surface",
    "label": "Rib width",
    "unit": "mm",
    "step": 0.1,
    "show_if": [
      {
        "surface_style": [
          "rib_lattice",
          "embossed"
        ]
      }
    ],
    "default": 2.0,
    "kind": "float",
    "min": 0.8,
    "max": 10
  },
  {
    "key": "emboss_depth",
    "group": "surface",
    "label": "Emboss depth",
    "unit": "mm",
    "step": 0.1,
    "show_if": [
      {
        "surface_style": [
          "embossed"
        ]
      }
    ],
    "max_field": "shell_thickness",
    "default": 0.8,
    "kind": "float",
    "min": 0.4,
    "max": 10
  },
  {
    "key": "edge_rounding",
    "group": "surface",
    "label": "Edge rounding",
    "unit": "mm",
    "step": 0.1,
    "show_if": [
      {
        "surface_style": [
          "perforated",
          "rib_lattice",
          "embossed"
        ]
      }
    ],
    "default": 0.5,
    "kind": "float",
    "min": 0,
    "max": 2
  },
  {
    "key": "interior_style",
    "group": "interior",
    "label": "Interior style",
    "unit": "",
    "default": "struts",
    "kind": "enum",
    "options": [
      {
        "value": "struts",
        "label": "Struts"
      },
      {
        "value": "solid",
        "label": "Solid"
      },
      {
        "value": "hollow",
        "label": "Hollow"
      }
    ]
  },
  {
    "key": "interior_cell_size",
    "group": "interior",
    "label": "Interior cell size",
    "unit": "mm",
    "step": 0.5,
    "show_if": [
      {
        "interior_style": [
          "struts"
        ]
      }
    ],
    "default": 12.0,
    "kind": "float",
    "min": 3,
    "max": 60
  },
  {
    "key": "strut_diameter",
    "group": "interior",
    "label": "Strut diameter",
    "unit": "mm",
    "step": 0.1,
    "show_if": [
      {
        "interior_style": [
          "struts"
        ]
      }
    ],
    "default": 2.0,
    "kind": "float",
    "min": 0.8,
    "max": 10
  },
  {
    "key": "blend_radius",
    "group": "modifiers",
    "label": "Blend radius",
    "unit": "mm",
    "step": 0.1,
    "show_if": [
      {
        "interior_style": [
          "struts"
        ]
      },
      {
        "surface_style": [
          "rib_lattice"
        ]
      }
    ],
    "default": 0.5,
    "kind": "float",
    "min": 0,
    "max": 5
  },
  {
    "key": "grading_mode",
    "group": "modifiers",
    "label": "Grading mode",
    "unit": "",
    "show_if": [
      {
        "surface_style": [
          "perforated",
          "rib_lattice",
          "embossed"
        ]
      },
      {
        "interior_style": [
          "struts"
        ]
      }
    ],
    "default": "off",
    "kind": "enum",
    "options": [
      {
        "value": "off",
        "label": "Off"
      },
      {
        "value": "from_skin",
        "label": "From skin"
      },
      {
        "value": "x",
        "label": "Along X"
      },
      {
        "value": "y",
        "label": "Along Y"
      },
      {
        "value": "z",
        "label": "Along Z"
      }
    ]
  },
  {
    "key": "grading_strength",
    "group": "modifiers",
    "label": "Grading strength",
    "unit": "%",
    "step": 1,
    "show_if": [
      {
        "surface_style": [
          "perforated",
          "rib_lattice",
          "embossed"
        ],
        "grading_mode": [
          "from_skin",
          "x",
          "y",
          "z"
        ]
      },
      {
        "interior_style": [
          "struts"
        ],
        "grading_mode": [
          "from_skin",
          "x",
          "y",
          "z"
        ]
      }
    ],
    "default": 50.0,
    "kind": "float",
    "min": 0,
    "max": 100
  },
  {
    "key": "grading_target",
    "group": "modifiers",
    "label": "Grading target",
    "unit": "",
    "show_if": [
      {
        "surface_style": [
          "perforated",
          "rib_lattice",
          "embossed"
        ],
        "grading_mode": [
          "from_skin",
          "x",
          "y",
          "z"
        ]
      },
      {
        "interior_style": [
          "struts"
        ],
        "grading_mode": [
          "from_skin",
          "x",
          "y",
          "z"
        ]
      }
    ],
    "default": "cell_size",
    "kind": "enum",
    "options": [
      {
        "value": "cell_size",
        "label": "Cell size"
      },
      {
        "value": "thickness",
        "label": "Thickness"
      },
      {
        "value": "both",
        "label": "Both"
      }
    ]
  },
  {
    "key": "grading_direction",
    "group": "modifiers",
    "label": "Grading direction",
    "unit": "",
    "show_if": [
      {
        "surface_style": [
          "perforated",
          "rib_lattice",
          "embossed"
        ],
        "grading_mode": [
          "from_skin",
          "x",
          "y",
          "z"
        ]
      },
      {
        "interior_style": [
          "struts"
        ],
        "grading_mode": [
          "from_skin",
          "x",
          "y",
          "z"
        ]
      }
    ],
    "default": "denser_at_start",
    "kind": "enum",
    "options": [
      {
        "value": "denser_at_start",
        "label": "Denser at start (near skin)"
      },
      {
        "value": "denser_at_end",
        "label": "Denser at end"
      }
    ]
  },
  {
    "key": "solid_base_thickness",
    "group": "modifiers",
    "label": "Solid base thickness (0 = off)",
    "unit": "mm",
    "step": 0.1,
    "default": 0.0,
    "kind": "float",
    "min": 0,
    "max": 5
  },
  {
    "key": "random_seed",
    "group": "modifiers",
    "label": "Random seed",
    "unit": "",
    "step": 1,
    "default": 1,
    "kind": "int",
    "min": 0,
    "max": 2147483647
  },
  {
    "key": "keep_edges",
    "group": "keep",
    "label": "Edges drawn by struts",
    "unit": "",
    "picked_in_viewport": true,
    "widget": "kept_edges",
    "default": [],
    "kind": "list"
  },
  {
    "key": "keep_width",
    "group": "keep",
    "label": "Edge strut width",
    "unit": "mm",
    "step": 0.1,
    "default": 2.0,
    "kind": "float",
    "min": 0.8,
    "max": 10
  },
  {
    "key": "bed_normal",
    "group": "printing",
    "label": "Bed face",
    "unit": "",
    "picked_in_viewport": true,
    "widget": "bed_face",
    "default": [
      0.0,
      0.0,
      -1.0
    ],
    "kind": "list"
  },
  {
    "key": "self_supporting",
    "group": "printing",
    "label": "Self-supporting struts",
    "unit": "",
    "show_if": [
      {
        "interior_style": [
          "struts"
        ]
      }
    ],
    "default": true,
    "kind": "bool"
  },
  {
    "key": "cell_stretch",
    "group": "printing",
    "label": "Cell stretch (print direction)",
    "unit": "×",
    "step": 0.1,
    "show_if": [
      {
        "interior_style": [
          "struts"
        ],
        "self_supporting": [
          "true"
        ]
      }
    ],
    "default": 2.0,
    "kind": "float",
    "min": 1.0,
    "max": 3.0
  },
  {
    "key": "overhang_angle",
    "group": "printing",
    "label": "Max overhang angle (from vertical)",
    "unit": "°",
    "step": 1,
    "show_if": [
      {
        "interior_style": [
          "struts"
        ],
        "self_supporting": [
          "true"
        ]
      },
      {
        "surface_style": [
          "perforated"
        ],
        "pointed_holes": [
          "true"
        ]
      }
    ],
    "default": 45.0,
    "kind": "float",
    "min": 30.0,
    "max": 60.0
  },
  {
    "key": "max_bridge",
    "group": "printing",
    "label": "Max bridge length",
    "unit": "mm",
    "step": 0.5,
    "show_if": [
      {
        "interior_style": [
          "struts"
        ],
        "self_supporting": [
          "true"
        ]
      }
    ],
    "default": 8.0,
    "kind": "float",
    "min": 0.0,
    "max": 20.0
  },
  {
    "key": "pointed_holes",
    "group": "printing",
    "label": "Pointed hole tops",
    "unit": "",
    "show_if": [
      {
        "surface_style": [
          "perforated"
        ]
      }
    ],
    "default": true,
    "kind": "bool"
  },
  {
    "key": "nozzle_diameter",
    "group": "print",
    "label": "Nozzle diameter",
    "unit": "mm",
    "step": 0.05,
    "default": 0.4,
    "kind": "float",
    "min": 0.2,
    "max": 1.0
  },
  {
    "key": "quality",
    "group": "print",
    "label": "Final quality",
    "unit": "",
    "default": "normal",
    "kind": "enum",
    "options": [
      {
        "value": "draft",
        "label": "Draft"
      },
      {
        "value": "normal",
        "label": "Normal"
      },
      {
        "value": "fine",
        "label": "Fine"
      }
    ]
  },
  {
    "key": "smoothness",
    "group": "print",
    "label": "Smoothness (final only)",
    "unit": "",
    "default": "off",
    "kind": "enum",
    "options": [
      {
        "value": "off",
        "label": "Off"
      },
      {
        "value": "smooth",
        "label": "Smooth (4 × triangles)"
      },
      {
        "value": "extra",
        "label": "Extra smooth (16 × triangles)"
      }
    ]
  },
  {
    "key": "cleanup",
    "group": "print",
    "label": "Clean-up (final only)",
    "unit": "",
    "default": "medium",
    "kind": "enum",
    "options": [
      {
        "value": "off",
        "label": "Off"
      },
      {
        "value": "light",
        "label": "Light"
      },
      {
        "value": "medium",
        "label": "Medium (recommended)"
      },
      {
        "value": "strong",
        "label": "Strong (fewest triangles)"
      }
    ]
  },
  {
    "key": "export_format",
    "group": "print",
    "label": "Export format",
    "unit": "",
    "default": "3mf",
    "kind": "enum",
    "options": [
      {
        "value": "3mf",
        "label": "3MF (compact, recommended)"
      },
      {
        "value": "stl",
        "label": "STL (binary)"
      }
    ]
  }
];

/** Combinations of values that are not allowed together (they describe an empty part). */
export const INVALID_COMBINATIONS: Partial<Record<SettingKey, string>>[] = [
  {
    "surface_style": "none",
    "interior_style": "hollow"
  }
];
