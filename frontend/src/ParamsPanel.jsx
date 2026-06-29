// Params editor for the selected node (spec §13). Renders one input per param,
// typed off the catalog default (checkbox for booleans, number for numbers, text
// for strings/lists/nullable). Save sends update_layer with the coerced params;
// the backend re-validates and broadcasts the result, so we never mutate locally.

import { useEffect, useState } from "react";
import { CATALOG } from "./catalog.js";
import { paramKind, coerce, toRaw } from "./paramTypes.js";

export default function ParamsPanel({ node, onSave, onClose, onDelete }) {
  // Per-param edit state, re-seeded whenever a different node is selected.
  const [form, setForm] = useState({});

  useEffect(() => {
    if (!node) return;
    const seeded = {};
    for (const [key, value] of Object.entries(node.data.params ?? {})) {
      seeded[key] = toRaw(value);
    }
    setForm(seeded);
  }, [node?.id]);

  if (!node) return null;

  const { type } = node.data;
  const catalogParams = CATALOG[type]?.params ?? node.data.params ?? {};
  const keys = Object.keys(node.data.params ?? {});

  function handleSave() {
    const params = {};
    for (const key of keys) {
      const kind = paramKind(catalogParams[key]);
      params[key] = coerce(kind, form[key]);
    }
    onSave(node.id, params);
  }

  return (
    <aside className="params">
      <div className="params__header">
        <span className="params__title">{type}</span>
        <span className="params__id">{node.id}</span>
      </div>

      {keys.length === 0 && <p className="params__empty">No parameters.</p>}

      <form
        className="params__form"
        onSubmit={(e) => {
          e.preventDefault();
          handleSave();
        }}
      >
        {keys.map((key) => {
          const kind = paramKind(catalogParams[key]);
          const value = form[key];
          return (
            <label key={key} className="params__field">
              <span className="params__label">{key}</span>
              {kind === "boolean" ? (
                <input
                  type="checkbox"
                  checked={!!value}
                  onChange={(e) => setForm((f) => ({ ...f, [key]: e.target.checked }))}
                />
              ) : (
                <input
                  type="text"
                  value={value ?? ""}
                  onChange={(e) => setForm((f) => ({ ...f, [key]: e.target.value }))}
                />
              )}
            </label>
          );
        })}

        <div className="params__actions">
          <button type="submit" className="params__save">Save</button>
          <button type="button" className="params__delete" onClick={() => onDelete(node.id)}>
            Delete
          </button>
          <button type="button" className="params__close" onClick={onClose}>Close</button>
        </div>
      </form>
    </aside>
  );
}
