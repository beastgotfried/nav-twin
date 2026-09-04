import React from "react";
import ReactDOM from "react-dom/client";
import { Theme } from "@astryxdesign/core";
import App from "./App";
import { sitaaraTheme } from "./theme";

// Astryx first, our own sheet last, so our tokens and overrides win the
// cascade. We consume the prebuilt dist (not the StyleX source build), which
// is why this needs no StyleX Vite plugin, no path aliases and no
// optimizeDeps juggling: the CSS is already compiled.
import "@astryxdesign/core/reset.css";
import "@astryxdesign/core/astryx.css";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Theme theme={sitaaraTheme} mode="dark">
      <App />
    </Theme>
  </React.StrictMode>,
);
