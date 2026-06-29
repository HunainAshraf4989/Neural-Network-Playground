// Modal showing the backend-generated PyTorch source with a copy-to-clipboard
// button. `result` is the `requestCode()` reply: {code} on success, {error} if
// the graph can't be generated (e.g. no input node).

import { useState } from "react";

export default function CodeModal({ result, onClose }) {
  const [copied, setCopied] = useState(false);
  const code = result?.code ?? "";

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="modal__backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal__header">
          <span className="modal__title">PyTorch code</span>
          <div className="modal__actions">
            {result?.code && (
              <button type="button" className="modal__copy" onClick={onCopy}>
                {copied ? "Copied!" : "Copy"}
              </button>
            )}
            <button type="button" className="modal__close" onClick={onClose} aria-label="Close">
              ×
            </button>
          </div>
        </div>
        {result?.error ? (
          <p className="modal__error">{result.error}</p>
        ) : (
          <pre className="modal__code">
            <code>{code}</code>
          </pre>
        )}
      </div>
    </div>
  );
}
