"""The canonical 19-class segmentation vocabulary, and a synonym map from
natural-language nouns (as VQA benchmarks and everyday questions phrase
them) onto it.

Per the training notebook (see scripts/demo_real.py's "no remapping
needed" note): the model's segmentation output index i is identity with
labels_all's raw class id i -- NOT the same order as
models/class_config.json's alphabetically-sorted classification-head
vocabulary. SEGMENTATION_CLASSES is the single source of truth for that
order; nothing else in this codebase should hardcode it a second time --
agent/planner.py's keyword fallback, scripts/demo_real.py, and every
eval/*.py script all import it from here.

Why this exists
-----------------
Earlier, scripts using this pipeline collapsed the model's 19 classes down
to a 4-bucket scheme (land/water/forest/building) before asking any
question. That made simple demo questions easy to resolve, but it also
meant any question about something outside those 4 buckets -- roads,
grass, residential areas, meadows, parks, heaths -- was refused outright,
even though the model actually predicts a class for exactly that thing.
NOUN_TO_CLASS below resolves natural-language nouns straight to the real
19-class vocabulary instead, so agent/planner.py's keyword fallback (and
anything built on it) can answer questions the 4-bucket scheme used to
reject.

A necessary consequence: agent/registry.py's TYPICAL_OBJECT_SIZE_M (which
drives the capability guardrail -- see agent/guardrail.py) previously only
recognised the illustrative bucket name "building". It now also carries
entries for "Urban fabric" and "Industrial or commercial units" -- the
real class names this vocabulary resolves built-up nouns to -- so the
guardrail keeps firing correctly instead of silently going inert once
callers stop passing the old 4-bucket scene.

Many-to-one AND many-to-many
--------------------------------
Most nouns resolve to exactly one segmentation class ("grass" ->
"Pastures"). A generic noun that legitimately spans several raw classes
-- "forest" could be Broad-leaved, Coniferous, or Mixed forest; "water"
could be Inland or Marine waters -- resolves to a LIST of class names
instead of picking one arbitrarily. This used to pick a single
"representative" class (e.g. "forest" -> "Mixed forest" only), which
silently returned 0 for any scene whose forest happened to be a different
subtype -- a real bug, not a hypothetical one (a scene that was 97%
Broad-leaved forest and 0% Mixed forest answered "0 hectares of forest").
evidence/ops.py's count/size/presence/adjacency all accept a list of
class ids for exactly this reason -- they union the classes into one
mask (`np.isin`) before measuring, so "how many forest patches" also
correctly merges adjacent pixels of different forest subtypes into one
connected patch rather than fragmenting them by raw class boundary.

Some nouns benchmarks ask about have NO match anywhere in this
vocabulary -- there is no dedicated road class, no sport/leisure class, no
airport/port/mine/dump/construction-site class, because BigEarthNet's
19-class reduction of CORINE Land Cover folds all of those into "Urban
fabric" (full ~44-class CORINE has separate classes for them; this
model's output does not). Where a mapping below routes such a noun to
"Urban fabric" anyway, that is a disclosed approximation -- the closest
real class, not a rediscovery of a class that doesn't exist.
"""

# --- The 19-class segmentation vocabulary -----------------------------------
# Index i is the model's raw predicted-mask class id i (see the module
# docstring). Do not alphabetize, reorder, or otherwise "clean up" this
# list -- see scripts/demo_real.py for what went wrong the last time this
# order was assumed rather than verified.

SEGMENTATION_CLASSES: list[str] = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

CLASS_NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(SEGMENTATION_CLASSES)}

assert len(SEGMENTATION_CLASSES) == 19


# --- Natural-language noun -> exact SEGMENTATION_CLASSES name ---------------
# Keys are lowercase, singular-or-plural as written. resolve_noun() below
# also tries stripping/adding a trailing "s", so most entries need only one
# form listed.

NOUN_TO_CLASS: dict[str, str | list[str]] = {
    # --- Urban fabric: residential/built-up references, plus the CORINE-only
    # classes BigEarthNet's 19-class reduction has no dedicated slot for at
    # all (road, sport/leisure, park) -- see the module docstring.
    "urban": "Urban fabric",
    "urban fabric": "Urban fabric",
    "urban area": "Urban fabric",
    "built-up area": "Urban fabric",
    "built up area": "Urban fabric",
    "building": "Urban fabric",
    "residential area": "Urban fabric",
    "residential building": "Urban fabric",
    "house": "Urban fabric",
    "road": "Urban fabric",  # approximation -- no dedicated road class exists, see docstring
    "street": "Urban fabric",
    "park": "Urban fabric",  # approximation -- no sport/leisure class exists, see docstring
    "pitch": "Urban fabric",
    "sports field": "Urban fabric",
    "sports pitch": "Urban fabric",

    # --- Industrial or commercial units ---
    "industrial area": "Industrial or commercial units",
    "industrial unit": "Industrial or commercial units",
    "commercial building": "Industrial or commercial units",
    "commercial unit": "Industrial or commercial units",
    "factory": "Industrial or commercial units",
    "warehouse": "Industrial or commercial units",

    # --- Arable land / farmland ---
    "farmland": "Arable land",
    "arable land": "Arable land",
    "cropland": "Arable land",
    "field": "Arable land",

    # --- Permanent crops ---
    "permanent crop": "Permanent crops",
    "orchard": "Permanent crops",
    "vineyard": "Permanent crops",

    # --- Pastures / grass / meadow -- the user's own example: grass, meadow,
    # and pasture all resolve to this one class.
    "pasture": "Pastures",
    "grass": "Pastures",
    "grass area": "Pastures",
    "grassland": "Pastures",
    "meadow": "Pastures",

    # --- Complex cultivation patterns ---
    "complex cultivation": "Complex cultivation patterns",
    "complex cultivation pattern": "Complex cultivation patterns",

    # --- Land principally occupied by agriculture ... ---
    "agricultural area": "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "agricultural land": "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "agriculture": "Land principally occupied by agriculture, with significant areas of natural vegetation",

    # --- Agro-forestry areas ---
    "agro-forestry": "Agro-forestry areas",
    "agroforestry": "Agro-forestry areas",

    # --- Forest: "forest"/"forests" is generic and spans all 3 real forest
    # classes -- resolves to all of them (see the module docstring). The
    # specific forest types still resolve to their own exact class alone.
    "forest": ["Broad-leaved forest", "Coniferous forest", "Mixed forest"],
    "broad-leaved forest": "Broad-leaved forest",
    "broadleaf forest": "Broad-leaved forest",
    "deciduous forest": "Broad-leaved forest",
    "coniferous forest": "Coniferous forest",
    "pine forest": "Coniferous forest",
    "mixed forest": "Mixed forest",
    "woodland": "Transitional woodland, shrub",
    "shrub": "Transitional woodland, shrub",
    "shrubland": "Transitional woodland, shrub",
    "scrub": "Transitional woodland, shrub",

    # --- Natural grassland and sparsely vegetated areas ---
    "natural grassland": "Natural grassland and sparsely vegetated areas",
    "sparsely vegetated area": "Natural grassland and sparsely vegetated areas",

    # --- Moors, heathland and sclerophyllous vegetation ---
    "heath": "Moors, heathland and sclerophyllous vegetation",
    "heathland": "Moors, heathland and sclerophyllous vegetation",
    "moor": "Moors, heathland and sclerophyllous vegetation",
    "moorland": "Moors, heathland and sclerophyllous vegetation",

    # --- Beaches, dunes, sands ---
    "beach": "Beaches, dunes, sands",
    "dune": "Beaches, dunes, sands",
    "sand": "Beaches, dunes, sands",

    # --- Wetlands ---
    "wetland": "Inland wetlands",
    "inland wetland": "Inland wetlands",
    "marsh": "Inland wetlands",
    "coastal wetland": "Coastal wetlands",

    # --- Water: "water"/"water area"/"water body" is generic -- resolves to
    # both Inland and Marine waters (see the module docstring). "sea"/
    # "ocean" still resolve to the correct "Marine waters" alone, and
    # "lake"/"river"/etc. to "Inland waters" alone.
    "water": ["Inland waters", "Marine waters"],
    "water area": ["Inland waters", "Marine waters"],
    "water body": ["Inland waters", "Marine waters"],
    "lake": "Inland waters",
    "river": "Inland waters",
    "pond": "Inland waters",
    "reservoir": "Inland waters",
    "canal": "Inland waters",
    "sea": "Marine waters",
    "ocean": "Marine waters",
    "marine water": "Marine waters",
}


def resolve_noun(phrase: str) -> str | list[str] | None:
    """Best-effort match of a natural-language noun phrase to one or more
    of SEGMENTATION_CLASSES's exact names (a list for a generic noun like
    "forest"/"water" that spans several real classes -- see the module
    docstring -- a plain str for everything else).

    Tries, in order: an exact NOUN_TO_CLASS lookup, the same lookup after
    stripping/adding a trailing "s" (covers plural/singular mismatches the
    map doesn't enumerate both forms of), then a NOUN_TO_CLASS-key-as-
    substring check (covers qualifiers the map doesn't enumerate, e.g.
    "small residential building" still contains "residential building").
    Returns None, never raises, if nothing matches -- callers decide how
    to treat that (agent/planner.py's keyword fallback raises PlannerError).
    """
    phrase = phrase.strip().lower()

    if phrase in NOUN_TO_CLASS:
        return NOUN_TO_CLASS[phrase]

    singular = phrase[:-1] if phrase.endswith("s") else phrase
    if singular in NOUN_TO_CLASS:
        return NOUN_TO_CLASS[singular]
    plural = f"{phrase}s"
    if plural in NOUN_TO_CLASS:
        return NOUN_TO_CLASS[plural]

    for noun, class_name in NOUN_TO_CLASS.items():
        if noun in phrase:
            return class_name

    return None
