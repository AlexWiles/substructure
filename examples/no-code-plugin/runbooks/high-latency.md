# High latency

1. Read the p99, not the average: the average hides it.
2. Check the cache hit rate; below 80% the database is doing the cache's work.
3. Flush and warm the cache: `cache-ctl warm --from-snapshot`.
4. Latency still high after a warm cache is a query problem — pull the slow
   query log before doing anything else.
