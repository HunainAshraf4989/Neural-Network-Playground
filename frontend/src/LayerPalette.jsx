// Sidebar of every catalog type, grouped by category (spec §13). Dragging a type
// onto the canvas triggers add_layer with that type's schema defaults; the drop
// coordinates are discarded (auto-layout owns positioning). Clicking a type adds
// it too, as a no-drag fallback.

import { groupedByCategory } from "./catalog.js";

const GROUPS = groupedByCategory();

export default function LayerPalette({ onAdd }) {
  return (
    <aside className="palette">
      <h2 className="palette__heading">Layers</h2>
      {GROUPS.map((group) => (
        <div key={group.category} className="palette__group">
          <div className="palette__group-title" style={{ color: group.color }}>
            {group.category}
          </div>
          <div className="palette__items">
            {group.types.map((type) => (
              <button
                key={type}
                type="button"
                className="palette__item"
                style={{ borderColor: group.color }}
                draggable
                onDragStart={(e) => {
                  // Carry the type so the canvas drop handler knows what to add.
                  e.dataTransfer.setData("application/nn-layer-type", type);
                  e.dataTransfer.effectAllowed = "copy";
                }}
                onClick={() => onAdd(type)}
                aria-label={`Add ${type}`}
                title={`Add ${type}`}
              >
                {type}
              </button>
            ))}
          </div>
        </div>
      ))}
    </aside>
  );
}
