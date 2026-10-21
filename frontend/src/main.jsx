import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import AuthGate from "./Auth.jsx";
import "./theme.css";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <AuthGate />
  </StrictMode>
);
