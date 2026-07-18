use xxhash_rust::xxh3::xxh3_64;

/// Stable key-to-shard assignment; must not change across versions
/// (persisted processor checkpoints are per shard).
pub(crate) fn shard_of(key: &str, shard_count: u32) -> usize {
    debug_assert!(shard_count > 0, "shard_count must be > 0");
    (xxh3_64(key.as_bytes()) % u64::from(shard_count)) as usize
}

pub(crate) fn in_shard(key: &str, shard_count: u32, shard_id: u32) -> bool {
    shard_of(key, shard_count) == shard_id as usize
}
