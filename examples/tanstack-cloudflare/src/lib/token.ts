import { createServerFn } from "@tanstack/react-start";

import { mintBrowserToken } from "../../substructure";

// Server function: mints the short-lived, identity-locked client token on the
// server (the API key never reaches the browser). Each page's route loader
// calls it, so the token is ready at first paint.
export const mintToken = createServerFn({ method: "POST" }).handler(() => mintBrowserToken());

export type BrowserSession = Awaited<ReturnType<typeof mintBrowserToken>>;
