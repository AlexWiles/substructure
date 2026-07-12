import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Chat } from "./chat";

createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <Chat />
    </StrictMode>,
);
