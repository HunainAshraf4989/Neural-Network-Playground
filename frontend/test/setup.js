import "@testing-library/jest-dom/vitest";

// React Flow measures the DOM; jsdom lacks these APIs. Minimal stubs so
// components that touch React Flow can mount in tests.
if (!global.ResizeObserver) {
  global.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

if (!global.DOMMatrixReadOnly) {
  global.DOMMatrixReadOnly = class {
    constructor() {
      this.m22 = 1;
    }
  };
}

if (!Element.prototype.getBoundingClientRect.__patched) {
  Element.prototype.getBoundingClientRect = function () {
    return { x: 0, y: 0, top: 0, left: 0, right: 0, bottom: 0, width: 100, height: 40 };
  };
  Element.prototype.getBoundingClientRect.__patched = true;
}
