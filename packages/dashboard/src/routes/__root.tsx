import { Outlet, createRootRouteWithContext } from '@tanstack/react-router'
import Header from '../components/Header'
import TanStackQueryProvider from '../integrations/tanstack-query/root-provider'

import '../styles.css'

import type { QueryClient } from '@tanstack/react-query'

interface MyRouterContext {
  queryClient: QueryClient
}

export const Route = createRootRouteWithContext<MyRouterContext>()({
  component: RootLayout,
})

function RootLayout() {
  return (
    <TanStackQueryProvider>
      <Header />
      <Outlet />
    </TanStackQueryProvider>
  )
}
