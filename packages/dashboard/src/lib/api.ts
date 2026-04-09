import { AdminClient } from "@substructure.ai/sdk/admin";

export const adminClient = new AdminClient({
    baseUrl: import.meta.env.VITE_API_BASE_URL ?? "",
});
