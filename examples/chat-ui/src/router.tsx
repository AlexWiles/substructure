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

// Layout route — Chat stays mounted when navigating between / and /sessions/$id
const chatLayout = createRoute({
  getParentRoute: () => rootRoute,
  id: "chat-layout",
  component: Chat,
});

const indexRoute = createRoute({
  getParentRoute: () => chatLayout,
  path: "/",
});

const sessionRoute = createRoute({
  getParentRoute: () => chatLayout,
  path: "sessions/$sessionId",
});

const routeTree = rootRoute.addChildren([
  chatLayout.addChildren([indexRoute, sessionRoute]),
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
