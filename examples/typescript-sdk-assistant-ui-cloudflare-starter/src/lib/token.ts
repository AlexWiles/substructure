import { createServerFn } from "@tanstack/react-start";

import { mintBrowserToken } from "../substructure";

// Mints the client token on the server so the API key never reaches the browser.
export const mintToken = createServerFn({ method: "POST" }).handler(() => mintBrowserToken());

export type BrowserSession = Awaited<ReturnType<typeof mintBrowserToken>>;
