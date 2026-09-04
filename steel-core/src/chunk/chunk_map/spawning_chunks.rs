//! Vanilla's natural-spawn candidate chunks and the 128-block player filter over them.
//!
//! Ports `ChunkMap.collectSpawningChunks`, `anyPlayerCloseEnoughForSpawning` and
//! `getPlayersCloseForSpawning` together with the `DistanceManager` state they read: the
//! `FixedPlayerDistanceChunkTracker(8)` that answers "how many chunks is this from a player", and
//! the chunk set that tracker keeps.
//!
//! Vanilla maintains the tracker incrementally, updating it from `ChunkMap.updatePlayerStatus` and
//! `ChunkMap.move`. Steel recomputes it from the live player list instead, into a
//! [`SpawningPlayers`] sample that answers every query. The answers are the same, because the
//! tracker propagates over a 3x3 neighborhood at a cost of one per step and keeps nothing past
//! its radius: its contents are exactly the union of the 17x17 chunk squares around the
//! player-occupied chunks. Sampling costs one pass over the players, which is the price of not
//! keeping a second body of incremental state in step with joins, moves, world changes and
//! game-mode switches.

use glam::DVec3;
use steel_registry::vanilla_game_rules::SPECTATORS_GENERATE_CHUNKS;

use super::{Arc, ChunkHolder, ChunkMap, ChunkPos, Entity, FxHashSet, Player};
use crate::world::natural_spawner::{
    INSCRIBED_SQUARE_SPAWN_DISTANCE_CHUNK, SPAWN_DISTANCE_BLOCK_SQUARED, SPAWN_DISTANCE_CHUNK,
};

/// Chunk distance reported for a chunk the spawn counter does not track.
///
/// Vanilla's `FixedPlayerDistanceChunkTracker` removes every node past its radius and gives its
/// backing map a default return value of `maxDistance + 2`, so a chunk nine or more chunks from
/// every player reads back as this rather than as its true distance. Callers compare against the
/// radius, never against this.
const SPAWN_CHUNK_DISTANCE_UNTRACKED: u32 = SPAWN_DISTANCE_CHUNK + 2;

/// What the chunk-distance pre-filter can settle on its own, vanilla's `TriState`.
///
/// Steel has no `TriState`, and the three cases are worth naming: close enough that no player
/// position can change the answer, far enough that none can either, or inside the ring where the
/// player list has to be walked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlayersNearby {
    /// Some player's spawn radius covers the whole chunk.
    Yes,
    /// No player is within [`SPAWN_DISTANCE_CHUNK`] chunks, so none can be within range.
    No,
    /// The chunk sits in the ring where the block distance decides.
    Undecided,
}

/// The players natural spawning consults, sampled once.
///
/// Stands in for vanilla's `ChunkMap.playerMap` plus the `DistanceManager` state built from it:
/// `playersPerChunk`'s key set becomes [`SpawningPlayers::occupied_chunks`], and the
/// `FixedPlayerDistanceChunkTracker(8)` over it becomes [`SpawningPlayers::chunk_distance`].
pub(crate) struct SpawningPlayers {
    /// Every player in the world, spectators included, as vanilla's `getAllPlayers` returns them.
    players: Vec<Arc<Player>>,
    /// The distinct chunks holding a player that counts toward the spawn-candidate set.
    occupied_chunks: Vec<ChunkPos>,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "vanilla spawn-candidate foundation; the NaturalSpawner port consumes it next, and \
                  the tests below already cover every method"
    )
)]
impl SpawningPlayers {
    /// Vanilla `FixedPlayerDistanceChunkTracker.getLevel` on the natural-spawn counter.
    ///
    /// The tracker's cost is one per step over a 3x3 neighborhood, so this is the Chebyshev chunk
    /// distance to the nearest occupied chunk, reported as [`SPAWN_CHUNK_DISTANCE_UNTRACKED`] once
    /// it passes [`SPAWN_DISTANCE_CHUNK`].
    #[must_use]
    fn chunk_distance(&self, pos: ChunkPos) -> u32 {
        let distance = self
            .occupied_chunks
            .iter()
            .map(|occupied| {
                occupied
                    .0
                    .x
                    .abs_diff(pos.0.x)
                    .max(occupied.0.y.abs_diff(pos.0.y))
            })
            .min()
            .unwrap_or(u32::MAX);

        if distance > SPAWN_DISTANCE_CHUNK {
            SPAWN_CHUNK_DISTANCE_UNTRACKED
        } else {
            distance
        }
    }

    /// Whether the spawn counter tracks this chunk, vanilla's `getSpawnCandidateChunks` membership.
    #[must_use]
    pub(crate) fn is_spawn_candidate(&self, pos: ChunkPos) -> bool {
        self.chunk_distance(pos) <= SPAWN_DISTANCE_CHUNK
    }

    /// Vanilla `DistanceManager.getNaturalSpawnChunkCount`, the size of the tracked chunk set.
    ///
    /// This is the mob cap's denominator, so it counts candidate chunks whether or not they are
    /// loaded: one lone player always contributes the full 17x17 square.
    #[must_use]
    pub(crate) fn natural_spawn_chunk_count(&self) -> usize {
        let mut tracked = FxHashSet::default();

        for occupied in &self.occupied_chunks {
            let min_x = occupied.0.x.saturating_sub_unsigned(SPAWN_DISTANCE_CHUNK);
            let max_x = occupied.0.x.saturating_add_unsigned(SPAWN_DISTANCE_CHUNK);
            let min_z = occupied.0.y.saturating_sub_unsigned(SPAWN_DISTANCE_CHUNK);
            let max_z = occupied.0.y.saturating_add_unsigned(SPAWN_DISTANCE_CHUNK);

            for x in min_x..=max_x {
                for z in min_z..=max_z {
                    tracked.insert(ChunkPos::new(x, z));
                }
            }
        }

        tracked.len()
    }

    /// Vanilla `DistanceManager.hasPlayersNearby`.
    #[must_use]
    pub(crate) fn has_players_nearby(&self, pos: ChunkPos) -> PlayersNearby {
        let distance = self.chunk_distance(pos);

        if distance <= INSCRIBED_SQUARE_SPAWN_DISTANCE_CHUNK {
            PlayersNearby::Yes
        } else if distance > SPAWN_DISTANCE_CHUNK {
            PlayersNearby::No
        } else {
            PlayersNearby::Undecided
        }
    }

    /// Vanilla `ChunkMap.anyPlayerCloseEnoughForSpawning`.
    ///
    /// The pre-filter's `Yes` is taken at its word, which is vanilla's behaviour and vanilla's
    /// quirk: a chunk five chunks from a spectator passes here while
    /// [`SpawningPlayers::any_player_close_enough_for_spawning_internal`] rejects it.
    #[must_use]
    pub(crate) fn any_player_close_enough_for_spawning(&self, pos: ChunkPos) -> bool {
        match self.has_players_nearby(pos) {
            PlayersNearby::Yes => true,
            PlayersNearby::No => false,
            PlayersNearby::Undecided => self.any_player_close_enough_for_spawning_internal(pos),
        }
    }

    /// Vanilla `ChunkMap.anyPlayerCloseEnoughForSpawningInternal`, which skips the pre-filter.
    #[must_use]
    pub(crate) fn any_player_close_enough_for_spawning_internal(&self, pos: ChunkPos) -> bool {
        self.players
            .iter()
            .any(|player| Self::player_is_close_enough_for_spawning(player, pos))
    }

    /// Vanilla `ChunkMap.getPlayersCloseForSpawning`, the per-player local mob cap's input.
    ///
    /// Only a definite `No` short-circuits: vanilla resolves the undecided answer to `true` here
    /// and then filters every player anyway, because the list has to name the players individually.
    #[must_use]
    pub(crate) fn players_close_for_spawning(&self, pos: ChunkPos) -> Vec<Arc<Player>> {
        if self.has_players_nearby(pos) == PlayersNearby::No {
            return Vec::new();
        }

        self.players
            .iter()
            .filter(|player| Self::player_is_close_enough_for_spawning(player, pos))
            .map(Arc::clone)
            .collect()
    }

    /// Vanilla `ChunkMap.playerIsCloseEnoughForSpawning`.
    #[must_use]
    fn player_is_close_enough_for_spawning(player: &Player, pos: ChunkPos) -> bool {
        if player.is_spectator() {
            return false;
        }

        Self::euclidean_distance_squared(pos, player.position())
            < f64::from(SPAWN_DISTANCE_BLOCK_SQUARED)
    }

    /// Vanilla `ChunkMap.euclideanDistanceSquared`: horizontal only, from the chunk's center.
    ///
    /// The `y` difference is dropped entirely, so this measures a cylinder rather than a sphere.
    #[must_use]
    fn euclidean_distance_squared(pos: ChunkPos, position: DVec3) -> f64 {
        let x_offset = Self::chunk_center_block_coord(pos.0.x) - position.x;
        let z_offset = Self::chunk_center_block_coord(pos.0.y) - position.z;
        x_offset * x_offset + z_offset * z_offset
    }

    /// Vanilla `SectionPos.sectionToBlockCoord(chunkCoord, 8)`, the chunk's center block column.
    #[must_use]
    fn chunk_center_block_coord(chunk_coord: i32) -> f64 {
        f64::from(chunk_coord) * 16.0 + 8.0
    }
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "vanilla spawn-candidate foundation; the NaturalSpawner port consumes it next, and \
                  the tests below already cover every method"
    )
)]
impl ChunkMap {
    /// Samples the players this world's natural spawning consults.
    ///
    /// Vanilla holds this state between ticks and so reads `spectators_generate_chunks` at every
    /// join, move and game-mode switch; sampling reads it once, here.
    #[must_use]
    pub(crate) fn spawning_players(&self) -> SpawningPlayers {
        let world = self.world_gen_context.world();
        let spectators_count = world.get_game_rule(&SPECTATORS_GENERATE_CHUNKS);
        let mut players = Vec::with_capacity(world.players.len());
        let mut occupied_chunks = Vec::new();

        world.players.iter_players(|_uuid, player| {
            // Vanilla `ChunkMap.skipPlayer`: a spectator stays out of the counter only when the
            // gamerule denies it, and the gamerule allows it by default.
            if spectators_count || !player.is_spectator() {
                let chunk = ChunkPos::from_entity_pos(player.position());
                if !occupied_chunks.contains(&chunk) {
                    occupied_chunks.push(chunk);
                }
            }

            players.push(Arc::clone(player));
            true
        });

        SpawningPlayers {
            players,
            occupied_chunks,
        }
    }

    /// Vanilla `ServerLevel.anyPlayerCloseEnoughForSpawning`, which samples the players itself.
    #[must_use]
    pub(crate) fn any_player_close_enough_for_spawning(&self, pos: ChunkPos) -> bool {
        self.spawning_players()
            .any_player_close_enough_for_spawning(pos)
    }

    /// Vanilla `ChunkMap.collectSpawningChunks`.
    ///
    /// Vanilla walks the tracker's chunk set and keeps the block-ticking ones; this walks the
    /// block-ticking snapshot and keeps the candidates. The set is the same either way — a chunk
    /// within [`SPAWN_DISTANCE_BLOCK_SQUARED`]'s radius of a non-spectator is necessarily within
    /// [`SPAWN_DISTANCE_CHUNK`] chunks of that player's own chunk — and walking the snapshot needs
    /// no map lookups. The order differs and does not matter: the caller shuffles.
    #[must_use]
    pub(crate) fn collect_spawning_chunks(
        &self,
        players: &SpawningPlayers,
    ) -> Vec<Arc<ChunkHolder>> {
        let snapshot = self.ticking_chunks.load();

        snapshot
            .block
            .iter()
            .filter(|chunk| {
                players.is_spawn_candidate(chunk.pos)
                    && players.any_player_close_enough_for_spawning_internal(chunk.pos)
            })
            .map(|chunk| Arc::clone(&chunk.holder))
            .collect()
    }
}
