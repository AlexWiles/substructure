import {
  createRouter,
  createRoute,
  createRootRoute,
  Outlet,
} from "@tanstack/react-router";
import { Chat } from "./routes/Chat";

const rootRoute = createRootRoute({
  component: () => <Outlet />,
});

const chatRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: Chat,
});

const routeTree = rootRoute.addChildren([chatRoute]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
