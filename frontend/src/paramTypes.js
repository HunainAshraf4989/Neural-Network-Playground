// Param value coercion between the catalog/JSON representation and the text
// inputs in the params panel. The backend stores param values verbatim (it only
// checks presence/unknown keys, never types), so it matters that "3" round-trips
// back to the number 3 and "[1,2]" back to the array [1, 2] before update_layer.
//
// Pure and dependency-free → exhaustively unit-tested in paramTypes.test.js.

// Classify a param by its catalog default value.
//   boolean → checkbox; number → numeric text; array → comma/JSON list;
//   null default (e.g. pooling stride) → nullable number; everything else string.
export function paramKind(defaultValue) {
  if (typeof defaultValue === "boolean") return "boolean";
  if (Array.isArray(defaultValue)) return "array";
  if (defaultValue === null) return "nullable";
  if (typeof defaultValue === "number") return "number";
  return "string";
}

// Value (from backend) → raw editable form for the input element.
export function toRaw(value) {
  if (typeof value === "boolean") return value;
  if (Array.isArray(value)) return value.join(",");
  if (value === null || value === undefined) return "";
  return String(value);
}

function toNumberOrRaw(raw) {
  const s = String(raw).trim();
  if (s === "") return raw;
  const n = Number(s);
  return Number.isFinite(n) ? n : raw;
}

// Raw form value → typed value to send in update_layer.
export function coerce(kind, raw) {
  switch (kind) {
    case "boolean":
      return !!raw;
    case "number":
      return toNumberOrRaw(raw);
    case "nullable": {
      const s = String(raw).trim();
      return s === "" ? null : toNumberOrRaw(s);
    }
    case "array":
      return parseList(raw);
    default:
      return raw == null ? "" : String(raw);
  }
}

// "[1,2,3]" or "1,2,3" → [1, 2, 3]. Numeric-looking items become numbers; others
// stay strings so the user can still type, say, a mode list if ever needed.
function parseList(raw) {
  const s = String(raw).trim();
  if (s === "") return [];
  if (s.startsWith("[")) {
    try {
      const parsed = JSON.parse(s);
      if (Array.isArray(parsed)) return parsed;
    } catch {
      /* fall through to comma parsing */
    }
  }
  return s
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part !== "")
    .map((part) => {
      const n = Number(part);
      return Number.isFinite(n) ? n : part;
    });
}
