use xxhash_rust::xxh3::xxh3_64;

/// Key-to-shard hash. Every runner must agree on it: a stream no shard claims
/// stalls until they do. Nothing is skipped or replayed — cursors are per
/// stream, not per shard. Masked to 63 bits so SQL shards it the same way.
pub(crate) fn shard_key(key: &str) -> u64 {
    xxh3_64(key.as_bytes()) & i64::MAX as u64
}

pub(crate) fn shard_of(key: &str, shard_count: u32) -> usize {
    debug_assert!(shard_count > 0, "shard_count must be > 0");
    (shard_key(key) % u64::from(shard_count)) as usize
}
