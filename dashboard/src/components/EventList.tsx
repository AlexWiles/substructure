import type { GroupedItem } from './events/index.ts'
import { ItemList } from './events/EventRows.tsx'

export function EventList({ items }: { items: GroupedItem[] }) {
  return <ItemList items={items} />
}
