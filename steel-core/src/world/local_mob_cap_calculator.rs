//! Vanilla's `LocalMobCapCalculator`, the per-player half of the mob cap.
//!
//! Ports `net.minecraft.world.level.LocalMobCapCalculator`. The global cap divides one category's
//! mob count by the whole spawnable-chunk area, so a large server can hit it with mobs nobody is
//! standing near; this second cap is per player, counting only the mobs whose chunk is within the
//! spawn radius of that player and refusing a spawn once **every** nearby player is already at
//! [`MobCategory::max_instances_per_chunk`] for the category.
//!
//! One divergence from vanilla, deliberate: vanilla holds a `ChunkMap` and asks it for the players
//! near a chunk, so a spawn tick walks the player list once per distinct chunk. Steel borrows the
//! tick's [`SpawningPlayers`] sample instead. The answers are identical — the sample is the same
//! player list vanilla would walk — and the memo below still keeps it to one filter pass per chunk.

use rustc_hash::FxHashMap;
use steel_registry::entity_type::MobCategory;

use super::{ChunkPos, Entity as _};
use crate::chunk::chunk_map::spawning_chunks::SpawningPlayers;

/// Vanilla `LocalMobCapCalculator`.
pub(crate) struct LocalMobCapCalculator<'a> {
    /// The tick's player sample, standing in for vanilla's `ChunkMap`.
    players: &'a SpawningPlayers,
    /// Memo of vanilla's `getPlayersNear`, holding entity ids rather than the players themselves
    /// because that is all the tally below is keyed by.
    players_near_chunk: FxHashMap<ChunkPos, Vec<i32>>,
    /// Vanilla's `playerMobCounts`, keyed by player entity id.
    player_mob_counts: FxHashMap<i32, MobCounts>,
}

impl<'a> LocalMobCapCalculator<'a> {
    /// Starts an empty calculator over `players`.
    pub(crate) fn new(players: &'a SpawningPlayers) -> Self {
        Self {
            players,
            players_near_chunk: FxHashMap::default(),
            player_mob_counts: FxHashMap::default(),
        }
    }

    /// Vanilla `getPlayersNear`, memoised per chunk.
    ///
    /// Takes the two fields it touches rather than `&mut self`, so a caller can hold the returned
    /// list while it mutates [`LocalMobCapCalculator::player_mob_counts`].
    fn players_near<'memo>(
        players: &SpawningPlayers,
        memo: &'memo mut FxHashMap<ChunkPos, Vec<i32>>,
        pos: ChunkPos,
    ) -> &'memo [i32] {
        memo.entry(pos)
            .or_insert_with(|| {
                players
                    .players_close_for_spawning(pos)
                    .iter()
                    .map(|player| player.id())
                    .collect()
            })
            .as_slice()
    }

    /// Vanilla `addMob`, which credits the mob to **every** player near its chunk.
    ///
    /// One mob counted against several players is the point: the cap asks whether any player has
    /// room, so a mob two players can both see has to fill both their allowances.
    pub(crate) fn add_mob(&mut self, pos: ChunkPos, category: MobCategory) {
        let nearby = Self::players_near(self.players, &mut self.players_near_chunk, pos);

        for player_id in nearby {
            self.player_mob_counts
                .entry(*player_id)
                .or_default()
                .add(category);
        }
    }

    /// Vanilla `canSpawn`: true as soon as one nearby player has room for the category.
    ///
    /// A chunk with no player near it answers `false`, which is vanilla's fall-through and not an
    /// oversight — this cap is only ever consulted for chunks the player filter already accepted.
    pub(crate) fn can_spawn(&mut self, category: MobCategory, pos: ChunkPos) -> bool {
        let nearby = Self::players_near(self.players, &mut self.players_near_chunk, pos);
        let player_mob_counts = &self.player_mob_counts;

        nearby.iter().any(|player_id| {
            player_mob_counts
                .get(player_id)
                .is_none_or(|counts| counts.can_spawn(category))
        })
    }
}

/// Vanilla's inner `MobCounts`: one player's tally, by category.
#[derive(Default)]
struct MobCounts {
    /// Mobs credited to this player so far, per category. A missing category counts as zero.
    counts: FxHashMap<MobCategory, i32>,
}

impl MobCounts {
    /// Vanilla `MobCounts.add`.
    fn add(&mut self, category: MobCategory) {
        *self.counts.entry(category).or_default() += 1;
    }

    /// Vanilla `MobCounts.canSpawn`.
    ///
    /// Vanilla's caller treats a player with no tally at all as having room, which is the same
    /// answer this gives for a zero count — except for `Misc`, whose cap is `-1`. That difference
    /// is unreachable: `Misc` is filtered out before any cap is consulted.
    fn can_spawn(&self, category: MobCategory) -> bool {
        self.counts.get(&category).copied().unwrap_or(0) < category.max_instances_per_chunk()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use glam::DVec3;
    use steel_registry::init_vanilla_registry;

    use super::{ChunkPos, LocalMobCapCalculator, MobCategory};
    use crate::behavior::init_behaviors;
    use crate::entity::Entity as _;
    use crate::player::{Player, ResetReason};
    use crate::test_support::{TestPlayerBuilder, fresh_test_world};
    use crate::world::World;

    /// Vanilla's cap for [`MobCategory::Creature`], the smallest of the spawning categories.
    const CREATURE_CAP: i32 = 10;

    /// The center of a chunk's block column, where the player filter measures distance from.
    fn chunk_center(chunk_x: i32, chunk_z: i32) -> DVec3 {
        DVec3::new(
            f64::from(chunk_x) * 16.0 + 8.0,
            64.0,
            f64::from(chunk_z) * 16.0 + 8.0,
        )
    }

    /// Builds a player standing at `position` and joins it to `world`.
    fn joined_player(
        world: &Arc<World>,
        name: &'static str,
        entity_id: i32,
        position: DVec3,
    ) -> Arc<Player> {
        let player = TestPlayerBuilder::new(Arc::clone(world), name, entity_id).build();
        assert!(
            player.try_set_position(position).is_ok(),
            "test player should be placed before joining"
        );
        assert!(world.add_player(Arc::clone(&player), ResetReason::InitialJoin));
        player
    }

    #[test]
    fn a_chunk_with_no_player_nearby_refuses_every_spawn() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("local_cap_no_player");
        let player = joined_player(&world, "LocalCapAlone", 1, chunk_center(0, 0));

        let players = world.chunk_map.spawning_players();
        let mut calculator = LocalMobCapCalculator::new(&players);

        assert!(calculator.can_spawn(MobCategory::Creature, ChunkPos::new(0, 0)));
        // Twenty chunks out, so the player filter yields an empty list and vanilla falls through.
        assert!(!calculator.can_spawn(MobCategory::Creature, ChunkPos::new(20, 0)));

        world.remove_player_for_world_change(&player);
    }

    #[test]
    fn a_player_at_the_category_cap_has_no_room_left_for_it() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("local_cap_at_cap");
        let player = joined_player(&world, "LocalCapFull", 1, chunk_center(0, 0));

        let players = world.chunk_map.spawning_players();
        let mut calculator = LocalMobCapCalculator::new(&players);
        let center = ChunkPos::new(0, 0);

        for _ in 0..CREATURE_CAP - 1 {
            calculator.add_mob(center, MobCategory::Creature);
        }
        assert!(calculator.can_spawn(MobCategory::Creature, center));

        calculator.add_mob(center, MobCategory::Creature);
        assert!(!calculator.can_spawn(MobCategory::Creature, center));
        // The tally is per category, so a monster is still welcome.
        assert!(calculator.can_spawn(MobCategory::Monster, center));

        world.remove_player_for_world_change(&player);
    }

    #[test]
    fn one_mob_is_credited_to_every_player_near_its_chunk() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("local_cap_shared");
        let center = ChunkPos::new(0, 0);
        // Both players stand in the candidate chunk, so both are near every mob added to it.
        let first = joined_player(&world, "LocalCapShareOne", 1, chunk_center(0, 0));
        let second = joined_player(&world, "LocalCapShareTwo", 2, chunk_center(0, 0));

        let players = world.chunk_map.spawning_players();
        let mut calculator = LocalMobCapCalculator::new(&players);

        for _ in 0..CREATURE_CAP {
            calculator.add_mob(center, MobCategory::Creature);
        }

        // Ten mobs fill both allowances. Crediting only one player would leave the other with room
        // and the chunk would still accept a spawn.
        assert!(!calculator.can_spawn(MobCategory::Creature, center));

        world.remove_player_for_world_change(&first);
        world.remove_player_for_world_change(&second);
    }

    #[test]
    fn a_mob_outside_a_players_range_leaves_their_allowance_alone() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("local_cap_scoped");
        let near = joined_player(&world, "LocalCapNear", 1, chunk_center(0, 0));
        let far = joined_player(&world, "LocalCapFar", 2, chunk_center(20, 0));

        let players = world.chunk_map.spawning_players();
        let mut calculator = LocalMobCapCalculator::new(&players);
        let far_chunk = ChunkPos::new(20, 0);

        for _ in 0..CREATURE_CAP {
            calculator.add_mob(far_chunk, MobCategory::Creature);
        }

        assert!(!calculator.can_spawn(MobCategory::Creature, far_chunk));
        // Two hundred and twenty blocks away, so none of those mobs count against this player.
        assert!(calculator.can_spawn(MobCategory::Creature, ChunkPos::new(0, 0)));

        world.remove_player_for_world_change(&near);
        world.remove_player_for_world_change(&far);
    }
}
