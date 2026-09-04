//! Vanilla's natural-spawning loop: the two mob caps, the crowding field's owner, the distances
//! the whole system is measured in, which mob a position is allowed to produce, and the per-tick
//! walk that turns all of that into live mobs.
//!
//! Ports `net.minecraft.world.level.NaturalSpawner` — `createState`,
//! `getFilteredSpawningCategories` and the `SpawnState` inner class over [`PotentialCalculator`]
//! and [`LocalMobCapCalculator`]; `mobsAt`, `getRandomSpawnMobAt`, `canSpawnMobAt` and
//! `isInNetherFortressBounds` over [`SpawnerTable`]; and `spawnForChunk`,
//! `spawnCategoryForChunk`, `spawnCategoryForPosition` and the four position tests under them.
//! [`tick_natural_spawns`] is the spawn phase of vanilla's `ServerChunkCache.tickChunks`, which
//! is where a tick enters this module, and [`spawn_mobs_for_chunk_generation`] is the separate
//! generation-time path that stocks a chunk before any player reaches it.
//!
//! Vanilla declares four distance constants together but reads only the last one: the tracker
//! radius reaches the spawn-chunk counter as a bare `8`, and the block threshold reaches the
//! per-player filter as a bare `16384.0`. Naming them keeps Steel's callers readable, so the tests
//! below pin the relationships between the names and the literals vanilla actually compares against.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "a few reads here serve spawn categories whose entity classes are not ported yet; \
                  the tests below already cover them"
    )
)]

use std::ptr;
use std::sync::Arc;

use glam::DVec3;
use rustc_hash::FxHashMap;
use steel_registry::biome::{SpawnCost, SpawnerData as BiomeSpawnerData};
use steel_registry::blocks::block_state_ext::BlockStateExt as _;
use steel_registry::entity_type::{EntityTypeRef, MobCategory};
use steel_registry::structure::{StructureSpawnBoundingBox, StructureSpawnerData};
use steel_registry::vanilla_biome_tags::BiomeTag;
use steel_registry::vanilla_game_rules::SPAWN_MOBS;
use steel_utils::WorldAabb;
use steel_utils::random::Random;
use steel_utils::random::legacy_random::LegacyRandom;
use steel_utils::types::Difficulty;

use super::{
    BiomeRef, BlockPos, ChunkPos, Identifier, REGISTRY, RegistryExt, World, vanilla_blocks,
};
use crate::chunk::chunk_holder::ChunkHolder;
use crate::chunk::chunk_map::spawning_chunks::SpawningPlayers;
use crate::chunk::heightmap::HeightmapType;
use crate::entity::spawn_placements;
use crate::entity::{
    ENTITIES, Entity as _, EntitySpawnReason, Mob, SharedEntity, SpawnGroupData, next_entity_id,
};
use crate::physics::collision::{
    WorldCollisionProvider, has_block_collision_in_level, has_collision,
};
use crate::world::ServerLevelAccessor as _;
use crate::world::local_mob_cap_calculator::LocalMobCapCalculator;
use crate::world::potential_calculator::PotentialCalculator;
use crate::world::signal_getter::is_redstone_conductor;
use crate::world::structure_manager::{
    all_structures_at, any_start_matching, has_structure_at, structure_has_piece_at,
    structure_start_contains,
};
use crate::worldgen::region::WorldGenRegion;

/// Chebyshev chunk radius around a player-occupied chunk within which mobs may spawn naturally.
///
/// Vanilla is the radius of `FixedPlayerDistanceChunkTracker(8)`, giving a 17x17 chunk square of
/// spawn candidates per occupied chunk.
pub(crate) const SPAWN_DISTANCE_CHUNK: u32 = 8;

/// Horizontal block radius within which a player enables spawning in a chunk.
///
/// Measured from the chunk's center column to the player's position, ignoring `y` entirely.
pub(crate) const SPAWN_DISTANCE_BLOCK: i32 = 128;

/// [`SPAWN_DISTANCE_BLOCK`] squared, so the filter compares squared distances.
///
/// This is the `16384.0` vanilla compares against; deriving it by multiplication keeps the two in
/// step and avoids a float constant in a `const` context.
pub(crate) const SPAWN_DISTANCE_BLOCK_SQUARED: i32 = SPAWN_DISTANCE_BLOCK * SPAWN_DISTANCE_BLOCK;

/// Chunk radius whose whole square lies inside the [`SPAWN_DISTANCE_BLOCK`] circle.
///
/// Vanilla computes it as `Mth.floor(8.0F / Mth.SQRT_OF_TWO)` — the square inscribed in the
/// [`SPAWN_DISTANCE_CHUNK`] circle. A chunk this close to an occupied chunk passes the block
/// distance test no matter which corner of their own chunk the player stands in, which lets the
/// proximity pre-filter answer without walking the player list.
pub(crate) const INSCRIBED_SQUARE_SPAWN_DISTANCE_CHUNK: u32 = 5;

/// Chunks one player contributes to the spawnable area, the global mob cap's divisor.
///
/// Vanilla calls it `MAGIC_NUMBER` and writes it as `(int)Math.pow(17.0, 2.0)`, which hides what it
/// is: the 17x17 candidate square of a single player. The global cap is therefore "this category's
/// per-chunk allowance, scaled by how many players' worth of area is loaded".
const SPAWN_CANDIDATE_SQUARE_CHUNKS: u32 = (2 * SPAWN_DISTANCE_CHUNK + 1).pow(2);

/// Squared distance to the nearest player inside which no natural spawn is allowed.
///
/// Vanilla writes the bare `576.0`, which is 24 blocks squared — the same 24 blocks the respawn
/// point is protected by, so the two exclusion zones are the same size and only the shape of the
/// comparison differs.
const PLAYER_SPAWN_EXCLUSION_DISTANCE_SQUARED: f64 = 576.0;

/// Block radius around the world's respawn point inside which no natural spawn is allowed.
const RESPAWN_EXCLUSION_DISTANCE: f64 = 24.0;

/// Group attempts vanilla makes per category per chunk per tick.
const SPAWN_GROUP_ATTEMPTS: i32 = 3;

/// Bound on each of the two draws that walk a group's members apart.
///
/// Vanilla's `nextInt(6) - nextInt(6)` gives a triangular step over -5..=5, so a member lands
/// within five blocks of its predecessor and most often within two.
const SPAWN_GROUP_WALK_SPREAD: i32 = 6;

/// Tick stride on which vanilla attempts the persistent categories.
///
/// Animals are a `persistent` category: they get one attempt every twentieth second rather than
/// one per tick, which is why a world fills with monsters far faster than with cows.
const PERSISTENT_SPAWN_INTERVAL: i64 = 400;

/// Vanilla `NaturalSpawner.getRoughBiome`, the *unfuzzed* noise biome at a block.
///
/// Vanilla reads it straight off the chunk it already holds, so it deliberately skips the fuzzing
/// [`World::biome_at`] applies. Steel's equivalent read is [`World::noise_biome_id`], which falls
/// back to the generator when the chunk is absent — harmless here, because every caller has already
/// established that the chunk is loaded.
fn rough_biome(world: &World, pos: BlockPos) -> Option<BiomeRef> {
    let biome_id = world.noise_biome_id(pos.x() >> 2, pos.y() >> 2, pos.z() >> 2)?;
    REGISTRY.biomes.by_id(usize::from(biome_id))
}

/// Vanilla `getRoughBiome(...).getMobSettings().getMobSpawnCost(type)`.
///
/// Most biomes declare no costs at all, in which case every type reads back `None` and the crowding
/// field is never consulted — that is vanilla's default and not a gap in the data.
fn mob_spawn_cost(
    world: &World,
    pos: BlockPos,
    entity_type: EntityTypeRef,
) -> Option<&'static SpawnCost> {
    rough_biome(world, pos)?.spawn_costs.get(&entity_type.key)
}

/// Vanilla `NaturalSpawner.createState`, the per-tick census of what is already alive.
///
/// Walks every entity the world can see and charges it against three tallies: the crowding field,
/// the global per-category count, and the per-player local cap. `spawnable_chunk_count` is the
/// caller's `DistanceManager.getNaturalSpawnChunkCount`, kept as a parameter exactly as vanilla
/// keeps it.
///
/// Two vanilla details that read as accidents but are not. A mob that is persistence-required or
/// requires custom persistence is skipped entirely, because a mob the player put there should not
/// suppress natural spawning; but the test is written so that *non-mob* entities always fall
/// through to be counted. And an entity in a chunk that is not loaded right now is not counted at
/// all, because vanilla's `ChunkGetter.query` only ever fires its callback for a live full chunk.
pub(crate) fn create_state<'players>(
    world: &World,
    spawnable_chunk_count: u32,
    players: &'players SpawningPlayers,
) -> SpawnState<'players> {
    let mut spawn_potential = PotentialCalculator::default();
    let mut mob_category_counts = FxHashMap::default();
    let mut local_mob_cap_calculator = LocalMobCapCalculator::new(players);

    for entity in world.entity_manager().get_accessible_entities() {
        let mob = entity.as_mob();
        if mob.is_some_and(|mob| mob.is_persistence_required() || mob.requires_custom_persistence())
        {
            continue;
        }

        let entity_type = entity.entity_type();
        let category = entity_type.mob_category;
        if category == MobCategory::Misc {
            continue;
        }

        let pos = entity.block_position();
        let chunk_pos = ChunkPos::from_block_pos(pos);
        if world.chunk_map.with_full_chunk(chunk_pos, |_| ()).is_none() {
            continue;
        }

        if let Some(cost) = mob_spawn_cost(world, pos, entity_type) {
            spawn_potential.add_charge(pos, cost.charge);
        }

        if mob.is_some() {
            local_mob_cap_calculator.add_mob(chunk_pos, category);
        }

        *mob_category_counts.entry(category).or_insert(0) += 1;
    }

    SpawnState {
        spawnable_chunk_count,
        mob_category_counts,
        spawn_potential,
        local_mob_cap_calculator,
        last_checked_pos: None,
        last_checked_type: None,
        last_charge: 0.0,
    }
}

/// Vanilla `NaturalSpawner.getFilteredSpawningCategories`.
///
/// The categories worth attempting at all this tick: those the two gamerule-derived flags allow and
/// that are under the global cap. `Misc` is excluded because it is not a spawning category —
/// vanilla filters it out of `SPAWNING_CATEGORIES` once, statically.
pub(crate) fn filtered_spawning_categories(
    state: &SpawnState<'_>,
    spawn_enemies: bool,
    spawn_persistent: bool,
) -> Vec<MobCategory> {
    MobCategory::ALL
        .into_iter()
        .filter(|category| *category != MobCategory::Misc)
        .filter(|category| {
            (spawn_enemies || category.is_friendly())
                && (spawn_persistent || !category.is_persistent())
                && state.can_spawn_for_category_global(*category)
        })
        .collect()
}

/// Vanilla's `NaturalSpawner.SpawnState`, one spawn tick's worth of accounting.
///
/// Holds both caps and the crowding field, and memoises the last cost it looked up so that the
/// `can_spawn` / `after_spawn` pair around a successful spawn costs one biome read rather than two.
pub(crate) struct SpawnState<'players> {
    /// Vanilla's `spawnableChunkCount`, the natural-spawn chunk count this tick.
    spawnable_chunk_count: u32,
    /// Live mobs per category, the global cap's numerator. A missing category counts as zero.
    mob_category_counts: FxHashMap<MobCategory, i32>,
    /// The crowding field every live mob has been charged into.
    spawn_potential: PotentialCalculator,
    /// The per-player cap.
    local_mob_cap_calculator: LocalMobCapCalculator<'players>,
    /// Position of the last [`SpawnState::can_spawn`] query, for the cost memo.
    last_checked_pos: Option<BlockPos>,
    /// Type of the last [`SpawnState::can_spawn`] query, for the cost memo.
    last_checked_type: Option<EntityTypeRef>,
    /// Charge the last query resolved to, valid only for that exact position and type.
    last_charge: f64,
}

impl SpawnState<'_> {
    /// Vanilla `SpawnState.canSpawn`, the crowding check for one candidate position.
    ///
    /// A type its biome declares no cost for always passes: vanilla's crowding field only thins the
    /// handful of mobs whose biome bothers to price them. Note the comparison is `<=`, so a spawn
    /// that lands exactly on the budget is allowed.
    fn can_spawn(&mut self, world: &World, entity_type: EntityTypeRef, test_pos: BlockPos) -> bool {
        self.last_checked_pos = Some(test_pos);
        self.last_checked_type = Some(entity_type);

        let Some(cost) = mob_spawn_cost(world, test_pos, entity_type) else {
            self.last_charge = 0.0;
            return true;
        };

        self.last_charge = cost.charge;
        self.spawn_potential
            .potential_energy_change(test_pos, cost.charge)
            <= cost.energy_budget
    }

    /// Vanilla `SpawnState.afterSpawn`, which books a mob that actually spawned into all three
    /// tallies.
    ///
    /// The memo is reused only when the position **and** the type both match the last
    /// [`SpawnState::can_spawn`] query, because vanilla's caller may have moved the mob between the
    /// two calls.
    fn after_spawn(&mut self, world: &World, entity_type: EntityTypeRef, pos: BlockPos) {
        let memo_applies = self.last_checked_pos == Some(pos)
            && self
                .last_checked_type
                .is_some_and(|last| ptr::eq(last, entity_type));

        let charge = if memo_applies {
            self.last_charge
        } else {
            mob_spawn_cost(world, pos, entity_type).map_or(0.0, |cost| cost.charge)
        };

        self.spawn_potential.add_charge(pos, charge);

        let category = entity_type.mob_category;
        *self.mob_category_counts.entry(category).or_insert(0) += 1;
        self.local_mob_cap_calculator
            .add_mob(ChunkPos::from_block_pos(pos), category);
    }

    /// Vanilla `SpawnState.getSpawnableChunkCount`.
    pub(crate) const fn spawnable_chunk_count(&self) -> u32 {
        self.spawnable_chunk_count
    }

    /// Vanilla `SpawnState.getMobCategoryCounts`, which hands out an unmodifiable view.
    ///
    /// A shared reference is already that view, so Steel needs no wrapper.
    pub(crate) const fn mob_category_counts(&self) -> &FxHashMap<MobCategory, i32> {
        &self.mob_category_counts
    }

    /// Vanilla `SpawnState.canSpawnForCategoryGlobal`, the server-wide cap.
    ///
    /// The allowance is the category's per-chunk figure scaled by how much spawnable area is
    /// loaded, in vanilla's integer arithmetic: a world with less than one player's square loaded
    /// rounds the allowance **down**, and at a small enough count down to zero.
    fn can_spawn_for_category_global(&self, category: MobCategory) -> bool {
        let max_mob_count = category.max_instances_per_chunk()
            * self.spawnable_chunk_count() as i32
            / SPAWN_CANDIDATE_SQUARE_CHUNKS as i32;

        self.mob_category_count(category) < max_mob_count
    }

    /// Vanilla `SpawnState.canSpawnForCategoryLocal`, the per-player cap.
    ///
    /// Vanilla also lets `SharedConstants.DEBUG_IGNORE_LOCAL_MOB_CAP` force this true. That flag is
    /// a compile-time `false` in every shipped build and Steel has no debug-flag facility to hang it
    /// on, so it is left out rather than added as a knob nothing can set.
    fn can_spawn_for_category_local(&mut self, category: MobCategory, chunk_pos: ChunkPos) -> bool {
        self.local_mob_cap_calculator.can_spawn(category, chunk_pos)
    }

    /// Live mobs counted for `category`, vanilla's `Object2IntMap.getInt` default of zero.
    fn mob_category_count(&self, category: MobCategory) -> i32 {
        self.mob_category_counts
            .get(&category)
            .copied()
            .unwrap_or(0)
    }
}

/// Vanilla `MobSpawnSettings.SpawnerData` and its `Weighted` wrapper, flattened into one row.
///
/// Vanilla's record is `SpawnerData(EntityType<?> type, int minCount, int maxCount)` held inside a
/// `Weighted<T>(T value, int weight)`. Steel's two source tables already store the weight beside the
/// counts, so this keeps all four together and [`is_same_spawner_data`] does by hand what the split
/// gave vanilla for free: equality that cannot see the weight.
///
/// The entity type stays an unresolved id on purpose. Resolving it here would mean dropping rows the
/// registry does not know, which would silently change the weighted total — one unknown mob in a
/// datapack would shift every other mob's odds in that table. The caller resolves only the row that
/// wins, so an unknown id costs its share of the rolls and spawns nothing, which is what vanilla
/// does one layer further down.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SpawnerEntry {
    /// Registry key of the mob this row spawns.
    pub(crate) entity_type: &'static Identifier,
    /// This row's share of the table's total weight.
    pub(crate) weight: i32,
    /// Smallest group this row spawns, vanilla `minCount`.
    pub(crate) min_count: i32,
    /// Largest group this row spawns, vanilla `maxCount`.
    pub(crate) max_count: i32,
}

/// Vanilla `SpawnerData.equals`, which does not see the weight.
///
/// `WeightedList.contains` compares `item.value().equals(value)`, and the weight lives on the
/// `Weighted` wrapper rather than on the value, so two rows naming the same mob with the same count
/// range match even when their weights differ. `canSpawnMobAt` depends on that: the group loop walks
/// the spawn position up to five blocks per member, so the table it re-reads is not always the table
/// the row was drawn from, and a mob a structure prices differently from its biome must still
/// validate.
fn is_same_spawner_data(row: SpawnerEntry, other: SpawnerEntry) -> bool {
    row.entity_type == other.entity_type
        && row.min_count == other.min_count
        && row.max_count == other.max_count
}

/// Vanilla `WeightedList<MobSpawnSettings.SpawnerData>`, borrowed from whichever table owns the rows.
///
/// The two arms are the two places 26.2 keeps a spawn list: a biome's per-category list and a
/// structure's `spawn_overrides` entry. Both are `&'static` registry data, so the table is `Copy` and
/// reading one allocates nothing — which matters because the spawn loop builds one per candidate
/// position, three group attempts deep.
///
/// Vanilla precomputes a selector — a flat index table below 64 total weight, a subtract-walk above —
/// and Steel always walks. A walk over the handful of rows any real table holds costs less than
/// building the index that would replace it, and both selectors consume exactly one bounded draw, so
/// the sequence of random numbers is identical either way.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SpawnerTable {
    /// A biome's `spawners[category]` list.
    Biome(&'static [BiomeSpawnerData]),
    /// A structure's `spawn_overrides[category].spawns` list.
    Structure(&'static [StructureSpawnerData]),
}

impl SpawnerTable {
    /// Vanilla `MobSpawnSettings.EMPTY_MOB_LIST`.
    ///
    /// Vanilla returns this shared instance whenever a category is absent from a biome, and an empty
    /// table is also what an *explicitly* empty structure override means — six of the eight
    /// categories ancient cities and trial chambers declare have no rows at all, which suppresses
    /// spawning inside them rather than falling through to the biome.
    const EMPTY: Self = Self::Biome(&[]);

    /// Rows in this table, vanilla `WeightedList.unwrap().size()`.
    const fn len(self) -> usize {
        match self {
            Self::Biome(rows) => rows.len(),
            Self::Structure(rows) => rows.len(),
        }
    }

    /// Vanilla `WeightedList.isEmpty`, which tests `selector == null`.
    ///
    /// Vanilla's selector is null exactly when the total weight is zero, so a table of all-zero
    /// weights is empty there and not here. Nothing observes the difference: the one caller that
    /// tests emptiness is [`Self::get_random`], and it answers `None` for a zero total anyway.
    const fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Projects one row, whichever list it lives in.
    ///
    /// The two source structs carry the same four fields under the same names, so this is the only
    /// place the arms differ and everything above it reads one row type.
    fn entry(self, index: usize) -> Option<SpawnerEntry> {
        match self {
            Self::Biome(rows) => rows.get(index).map(|row| SpawnerEntry {
                entity_type: &row.entity_type,
                weight: row.weight,
                min_count: row.min_count,
                max_count: row.max_count,
            }),
            Self::Structure(rows) => rows.get(index).map(|row| SpawnerEntry {
                entity_type: &row.entity_type,
                weight: row.weight,
                min_count: row.min_count,
                max_count: row.max_count,
            }),
        }
    }

    /// Every row in declaration order, which is the order both selectors consume weights in.
    ///
    /// The `filter_map` cannot drop a row: every index below [`Self::len`] projects, and it is only
    /// how a fallible projection composes into an infallible walk.
    fn rows(self) -> impl Iterator<Item = SpawnerEntry> {
        (0..self.len()).filter_map(move |index| self.entry(index))
    }

    /// Vanilla `WeightedRandom.getTotalWeight`, which sums into a `long`.
    ///
    /// Vanilla throws above `Integer.MAX_VALUE`; this saturates instead. Weights are non-negative
    /// small integers by codec, so reaching the bound would take on the order of two billion of
    /// them, and a panic here would take down the tick that is merely asking what could spawn.
    fn total_weight(self) -> i32 {
        let total: i64 = self.rows().map(|row| i64::from(row.weight)).sum();

        i32::try_from(total).unwrap_or(i32::MAX)
    }

    /// Vanilla `WeightedList.getRandom`, the subtract-walk over one bounded draw.
    ///
    /// A zero total answers `None` — vanilla's null selector — and so would a negative one, which
    /// the codec's non-negative weights make unreachable. Falling out of the loop is vanilla's
    /// trailing `Optional.empty()`, equally unreachable while the bound is the real sum.
    fn get_random(self, random: &mut impl Random) -> Option<SpawnerEntry> {
        let total = self.total_weight();
        if total <= 0 {
            return None;
        }

        let mut selection = random.next_i32_bounded(total);
        for row in self.rows() {
            selection -= row.weight;
            if selection < 0 {
                return Some(row);
            }
        }

        None
    }

    /// Vanilla `WeightedList.contains`, and therefore weight-insensitive.
    ///
    /// See [`is_same_spawner_data`] for why that is load-bearing rather than an oversight.
    fn contains(self, entry: SpawnerEntry) -> bool {
        self.rows().any(|row| is_same_spawner_data(row, entry))
    }
}

/// The nether fortress's registry key, vanilla `BuiltinStructures.FORTRESS`.
const FORTRESS: Identifier = Identifier::vanilla_static("fortress");

/// Chance a water-ambient attempt is thrown away in a river, vanilla's `0.98F`.
///
/// Vanilla applies it in `getRandomSpawnMobAt` rather than in the cap, so a river still counts its
/// squid against the water-ambient cap — it just almost never gets to add one.
const WATER_AMBIENT_SKIP_CHANCE: f32 = 0.98;

/// Vanilla `MobSpawnSettings.getMobs`, over Steel's generated biome table.
///
/// Vanilla keys the map by `MobCategory` and answers `EMPTY_MOB_LIST` for an absent key;
/// `steel-registry/build/biomes.rs` emits the category as its lowercase name, which is exactly
/// [`MobCategory::name`], so the lookup is the same lookup one string away.
fn biome_spawners(biome: BiomeRef, category: MobCategory) -> SpawnerTable {
    biome
        .spawners
        .get(category.name())
        .map_or(SpawnerTable::EMPTY, |rows| {
            SpawnerTable::Biome(rows.as_slice())
        })
}

/// Vanilla `NetherFortressStructure.FORTRESS_ENEMIES`, read out of the registry instead.
///
/// Vanilla hardcodes this list in the structure class and *also* ships it as the fortress's
/// `spawn_overrides` entry for `monster`; the two are identical for vanilla data, and the loader
/// that reads the JSON is the one Steel already has. Reading it means a datapack that retunes
/// fortress spawning retunes both paths together, which is the behaviour a server operator expects
/// and the opposite of what copying the hardcoded list would give.
///
/// An absent structure or an absent category answers empty, which for the only caller —
/// [`mobs_at`]'s fortress branch, reached only when a fortress is actually present — means the
/// fortress declines to override and the position spawns nothing rather than falling through.
fn fortress_enemies(category: MobCategory) -> SpawnerTable {
    REGISTRY
        .structures
        .by_key(&FORTRESS)
        .and_then(|fortress| {
            fortress
                .spawn_overrides
                .iter()
                .find(|entry| entry.category == category.name())
        })
        .map_or(SpawnerTable::EMPTY, |entry| {
            SpawnerTable::Structure(entry.spawns.as_slice())
        })
}

/// Vanilla `ChunkGenerator.getMobsAt`, the `spawn_overrides` scan.
///
/// Walks every structure the position's chunk references and answers the first one that both
/// declares this category and encloses the position in the box its override names. `Piece` tests the
/// piece boxes so the gaps inside a structure are not claimed; `Full` tests the whole start's box so
/// they are. Vanilla folds the scan into a flag and keeps going; stopping at the first match is the
/// same answer because both box tests read only the position and the start.
///
/// Iteration order is the map's, which vanilla leaves as undefined as Steel does — `ChunkAccess`
/// builds both structure maps with `Maps.newHashMap()`. Two structures overlapping a position and
/// both overriding the same category is therefore already unspecified upstream, so `FxHashMap`'s
/// order needs no defending.
///
/// An override with no rows is a match, not a miss. Ancient cities and trial chambers declare most
/// of the eight categories with an empty `spawns` list precisely to suppress spawning inside
/// themselves, so this must answer `Some(EMPTY)` there and let the biome go unread.
fn structure_override_at(
    world: &World,
    pos: BlockPos,
    category: MobCategory,
) -> Option<SpawnerTable> {
    all_structures_at(world, pos)
        .into_iter()
        .find_map(|(structure, origins)| {
            let override_data = structure
                .spawn_overrides
                .iter()
                .find(|entry| entry.category == category.name())?;

            let in_override_box = match override_data.bounding_box {
                StructureSpawnBoundingBox::Piece => {
                    any_start_matching(world, &structure.key, &origins, |start| {
                        structure_has_piece_at(pos, start)
                    })
                }
                StructureSpawnBoundingBox::Full => {
                    any_start_matching(world, &structure.key, &origins, |start| {
                        structure_start_contains(pos, start)
                    })
                }
            };

            in_override_box.then_some(SpawnerTable::Structure(override_data.spawns.as_slice()))
        })
}

/// Vanilla `NaturalSpawner.mobsAt`, all three branches.
///
/// Fortress first, then the generic `spawn_overrides` scan, then the biome. The order is vanilla's
/// and it matters: a fortress is also a structure with a `monster` override, and the fortress branch
/// tests a *different* box — the whole start's, plus nether bricks underfoot — so a position inside
/// the fortress's outline but outside its pieces takes the fortress list here where the generic scan
/// would have declined it.
///
/// `biome` is the caller's already-fuzzed read, threaded through so the spawn loop reads the biome
/// once per position instead of once per branch. `None` means both "the caller has not read it" and
/// "the caller read it and there was no biome to read", and re-reading answers identically either
/// way — [`World::biome_at`] has no side effects and the fuzz is a pure function of the position.
/// Deferring the read until both structure branches miss is faithful for the same reason.
///
/// An unloaded position is not a special case. Vanilla's `getNoiseBiome` passes `load = false` and
/// falls back to `getUncachedNoiseBiome`, which asks the generator's biome source directly rather
/// than generating anything; Steel's [`World::biome_at`] does the same, so a position outside every
/// loaded chunk reads the biome the generator would have written there. `SpawnerTable::EMPTY` is
/// only reached when that read cannot be resolved to a registered biome at all.
pub(crate) fn mobs_at(
    world: &World,
    category: MobCategory,
    pos: BlockPos,
    biome: Option<BiomeRef>,
) -> SpawnerTable {
    if is_in_nether_fortress_bounds(world, pos, category) {
        return fortress_enemies(category);
    }

    if let Some(table) = structure_override_at(world, pos, category) {
        return table;
    }

    biome
        .or_else(|| world.biome_at(pos))
        .map_or(SpawnerTable::EMPTY, |biome| biome_spawners(biome, category))
}

/// Vanilla `NaturalSpawner.isInNetherFortressBounds`.
///
/// Three terms, cheapest first, exactly as vanilla orders them: the category, the block below, and
/// only then the structure lookup that has to reach into neighboring chunks. The block test is why
/// a fortress's courtyards and the soul sand around them do not spawn blazes — the whole-start box
/// covers them, the nether bricks do not.
pub(crate) fn is_in_nether_fortress_bounds(
    world: &World,
    pos: BlockPos,
    category: MobCategory,
) -> bool {
    category == MobCategory::Monster
        && world.get_block_state(pos.below()).get_block() == &vanilla_blocks::NETHER_BRICKS
        && has_structure_at(world, pos, &FORTRESS)
}

/// Vanilla `NaturalSpawner.getRandomSpawnMobAt`.
///
/// The river clause is vanilla's one hand-placed thumb on the scales: rivers share the ocean's
/// water-ambient list, and without it a river would be as full of squid as an ocean. Note the guard
/// order — the float is drawn only once the category and the tag both match, so a run of positions
/// that are not water-ambient consumes no randomness at all and Steel's draw sequence stays aligned
/// with vanilla's.
///
/// The biome is read here and handed to [`mobs_at`], which is vanilla's structure too: one fuzzed
/// read serves both the tag test and the table.
pub(crate) fn random_spawn_mob_at(
    world: &World,
    category: MobCategory,
    pos: BlockPos,
    random: &mut impl Random,
) -> Option<SpawnerEntry> {
    let biome = world.biome_at(pos);

    if category == MobCategory::WaterAmbient
        && biome.is_some_and(|biome| biome.has_tag(&BiomeTag::REDUCE_WATER_AMBIENT_SPAWNS))
        && random.next_f32() < WATER_AMBIENT_SKIP_CHANCE
    {
        return None;
    }

    mobs_at(world, category, pos, biome).get_random(random)
}

/// Vanilla `NaturalSpawner.canSpawnMobAt`.
///
/// Re-asks the tables at the position the group has actually drifted to and checks the row is still
/// offered there. Passing `None` for the biome is vanilla's `mobsAt(..., pos)` with no cached read:
/// the drifted position may well be in a different biome than the one the row was drawn from, so
/// caching the earlier read would be the bug rather than the optimization.
pub(crate) fn can_spawn_mob_at(
    world: &World,
    category: MobCategory,
    pos: BlockPos,
    entry: SpawnerEntry,
) -> bool {
    mobs_at(world, category, pos, None).contains(entry)
}

/// Vanilla `NaturalSpawner.getRandomPosWithin`.
///
/// The `y` is drawn over the *whole* column, from the world floor to one above the highest
/// non-air block, which is why deep caves get their spawn attempts from the same loop that
/// serves the surface.
///
/// The heightmap read looks like it is missing vanilla's `+ 1` and is not. Vanilla asks
/// `chunk.getHeight`, which returns the highest occupied `y`; Steel's [`World::height_at`] is
/// the underlying `getFirstAvailable`, which is already that value plus one. So
/// `topEmptyY = getHeight + 1` and `height_at` are the same number.
fn get_random_pos_within(world: &World, chunk_pos: ChunkPos, random: &mut impl Random) -> BlockPos {
    let x = chunk_pos.0.x * 16 + random.next_i32_bounded(16);
    let z = chunk_pos.0.y * 16 + random.next_i32_bounded(16);
    let top_empty_y = world
        .height_at(HeightmapType::WorldSurface, x, z)
        .unwrap_or_else(|| world.get_min_y());
    let y = random.next_i32_between(world.get_min_y(), top_empty_y);
    BlockPos::new(x, y, z)
}

/// Vanilla `NaturalSpawner.isRightDistanceToPlayerAndSpawnPoint`.
///
/// Three refusals: too close to the nearest player, too close to the world's respawn point, or
/// in a chunk that is neither the chunk the attempt started in nor one entities may be added to.
/// The `24.0` test is a sphere around the respawn block's center and the `576.0` is that same
/// radius squared applied to the player — vanilla writes the two differently but they are the
/// same distance.
///
/// Steel reads the respawn point from this world's own level data rather than from a
/// server-global field, then keeps vanilla's dimension comparison. When a world has never had a
/// respawn point set, `respawn_data_or_local` answers with that world's spawn, so the dimension
/// matches and the 24-block exclusion applies — which is what vanilla does for the overworld and
/// a harmless no-op elsewhere.
fn is_right_distance_to_player_and_spawn_point(
    world: &World,
    origin_chunk: ChunkPos,
    pos: BlockPos,
    nearest_player_distance_sqr: f64,
) -> bool {
    if nearest_player_distance_sqr <= PLAYER_SPAWN_EXCLUSION_DISTANCE_SQUARED {
        return false;
    }

    let respawn_data = {
        let level_data = world.level_data.read();
        level_data.data().respawn_data_or_local(&world.key)
    };
    if respawn_data.dimension() == &world.key {
        let respawn_center = respawn_data.pos().get_center();
        let candidate = DVec3::new(
            f64::from(pos.x()) + 0.5,
            f64::from(pos.y()),
            f64::from(pos.z()) + 0.5,
        );
        let respawn_center = DVec3::new(respawn_center.0, respawn_center.1, respawn_center.2);
        if respawn_center.distance_squared(candidate)
            < RESPAWN_EXCLUSION_DISTANCE * RESPAWN_EXCLUSION_DISTANCE
        {
            return false;
        }
    }

    let chunk_pos = ChunkPos::from_block_pos(pos);
    chunk_pos == origin_chunk || can_spawn_entities_in_chunk(world, chunk_pos)
}

/// Vanilla `ServerLevel.canSpawnEntitiesInChunk`.
///
/// Both of vanilla's terms: the chunk must be entity-ticking, and its whole footprint must lie
/// inside the world border.
fn can_spawn_entities_in_chunk(world: &World, chunk_pos: ChunkPos) -> bool {
    if !world
        .chunk_map
        .is_entity_ticking_full_chunk_loaded(chunk_pos)
    {
        return false;
    }

    let border = world.world_border_snapshot();
    let min_x = f64::from(chunk_pos.0.x * 16);
    let min_z = f64::from(chunk_pos.0.y * 16);
    border.is_within_bounds_with_margin(min_x, min_z, 0.0)
        && border.is_within_bounds_with_margin(min_x + 15.0, min_z + 15.0, 0.0)
}

/// Vanilla `NaturalSpawner.isValidSpawnPositionForType`, whose real name misspells "Position" —
/// search Mojang's source for `isValidSpawnPos` to find it.
///
/// Seven terms in vanilla's order, and the order is load-bearing: the two cheap field reads and
/// the distance comparison come before the table re-read, and the collision sweep comes last.
fn is_valid_spawn_position_for_type(
    world: &Arc<World>,
    category: MobCategory,
    entry: SpawnerEntry,
    entity_type: EntityTypeRef,
    pos: BlockPos,
    nearest_player_distance_sqr: f64,
    random: &mut impl Random,
) -> bool {
    let despawn_distance = f64::from(entity_type.mob_category.despawn_distance());
    entity_type.mob_category != MobCategory::Misc
        && (entity_type.can_spawn_far_from_player
            || nearest_player_distance_sqr <= despawn_distance * despawn_distance)
        && entity_type.summonable
        && can_spawn_mob_at(world, category, pos, entry)
        && spawn_placements::is_spawn_position_ok(entity_type, world, pos)
        && spawn_placements::check_spawn_rules(
            entity_type,
            world,
            EntitySpawnReason::Natural,
            pos,
            random,
        )
        && !has_collision(
            &WorldCollisionProvider::new(world),
            WorldAabb::entity_box(
                f64::from(pos.x()) + 0.5,
                f64::from(pos.y()),
                f64::from(pos.z()) + 0.5,
                f64::from(entity_type.dimensions.half_width()),
                f64::from(entity_type.dimensions.height),
            ),
        )
}

/// Vanilla `NaturalSpawner.getMobForSpawn`.
///
/// `None` covers both of vanilla's failure paths and one Steel-only path. Vanilla's two — a type
/// whose factory declines to build, and a type that builds into something that is not a `Mob` —
/// stay at `warn`, because both mean a registered type is broken.
///
/// The Steel-only path is a type with **no registered factory at all**, which is every entity
/// class the port has not reached yet. During the port that is the ordinary case rather than an
/// error: the overworld's `monster` tables name zombies and skeletons on every draw, so warning
/// here would emit thousands of lines a second and drown the log the acceptance test reads. It
/// logs at `debug` and skips, which is the same control flow with an honest severity.
fn get_mob_for_spawn(world: &Arc<World>, entity_type: EntityTypeRef) -> Option<SharedEntity> {
    if !ENTITIES.has_factory(entity_type) {
        log::debug!(
            "No entity factory registered yet for {}; skipping natural spawn",
            entity_type.key
        );
        return None;
    }

    let Some(entity) = ENTITIES.create(
        entity_type,
        next_entity_id(),
        DVec3::ZERO,
        Arc::downgrade(world),
    ) else {
        log::warn!("Can't spawn entity of type: {}", entity_type.key);
        return None;
    };

    if entity.as_mob().is_none() {
        log::warn!("Can't spawn entity of type: {}", entity_type.key);
        return None;
    }

    Some(entity)
}

/// Vanilla `NaturalSpawner.isValidPositionForMob`.
///
/// Runs on a built and positioned mob, so it can ask the mob's own overrides rather than the
/// type's placement row. The first clause reads inside out: a mob far from every player is only
/// refused if it is the kind that despawns out there.
fn is_valid_position_for_mob(
    world: &Arc<World>,
    mob: &dyn Mob,
    nearest_player_distance_sqr: f64,
) -> bool {
    let despawn_distance = f64::from(mob.entity_type().mob_category.despawn_distance());
    (nearest_player_distance_sqr <= despawn_distance * despawn_distance
        || !mob.remove_when_far_away(nearest_player_distance_sqr))
        && mob.check_spawn_rules(world, EntitySpawnReason::Natural)
        && mob.check_spawn_obstruction(world)
}

/// Vanilla `Entity.snapTo`, for a mob that has been built but not yet added to the world.
///
/// Returns whether the placement committed. Vanilla cannot fail here, but Steel's position write
/// goes through the entity's level callback and can refuse, and a refusal must abandon the spawn:
/// the mob was created at the origin and would otherwise be added there.
fn snap_to(entity: &SharedEntity, position: DVec3, yaw: f32, pitch: f32) -> bool {
    if let Err(error) = entity.try_set_position(position) {
        log::warn!(
            "Failed to place naturally spawned entity {}: {error}",
            entity.id()
        );
        return false;
    }
    entity.set_rotation((yaw, pitch));
    entity.set_old_position_to_current();
    true
}

/// Vanilla `NaturalSpawner.spawnCategoryForPosition`.
///
/// Three group attempts from the same starting column, each walking up to
/// [`SPAWN_GROUP_WALK_SPREAD`] blocks per member in `x` and `z` while holding `y` fixed. Two
/// details that look like bugs and are vanilla:
///
/// `max` is rolled twice. It starts as a 1-4 draw so that a position offering no mobs still
/// costs a bounded number of attempts, and is then *overwritten* by the winning row's own count
/// range the first time a row is drawn — mid-loop, with `ll` already advanced.
///
/// The walk keeps `y` from the original draw, so a group can drift into a wall or over a cliff;
/// the per-candidate tests are what stop it, not the walk.
fn spawn_category_for_position(
    world: &Arc<World>,
    category: MobCategory,
    origin_chunk: ChunkPos,
    start: BlockPos,
    state: &mut SpawnState<'_>,
    random: &mut impl Random,
) {
    let state_block = world.get_block_state(start);
    if is_redstone_conductor(world.as_ref(), state_block, start) {
        return;
    }

    let y_start = start.y();
    let mut cluster_size = 0_i32;

    for _ in 0..SPAWN_GROUP_ATTEMPTS {
        let mut x = start.x();
        let mut z = start.z();
        let mut current: Option<(SpawnerEntry, EntityTypeRef)> = None;
        let mut group_data = None;
        let mut max = random.next_f32().mul_add(4.0, 0.0).ceil() as i32;
        let mut group_size = 0_i32;

        let mut member = 0_i32;
        while member < max {
            member += 1;
            x += random.next_i32_bounded(SPAWN_GROUP_WALK_SPREAD)
                - random.next_i32_bounded(SPAWN_GROUP_WALK_SPREAD);
            z += random.next_i32_bounded(SPAWN_GROUP_WALK_SPREAD)
                - random.next_i32_bounded(SPAWN_GROUP_WALK_SPREAD);
            let pos = BlockPos::new(x, y_start, z);
            let center = DVec3::new(f64::from(x) + 0.5, f64::from(y_start), f64::from(z) + 0.5);

            let Some(nearest_player) =
                world.nearest_player(center, -1.0, |player| !player.is_spectator())
            else {
                continue;
            };
            let nearest_player_distance_sqr = nearest_player.position().distance_squared(center);
            if !is_right_distance_to_player_and_spawn_point(
                world,
                origin_chunk,
                pos,
                nearest_player_distance_sqr,
            ) {
                continue;
            }

            if current.is_none() {
                let Some(entry) = random_spawn_mob_at(world, category, pos, random) else {
                    break;
                };
                let Some(entity_type) = REGISTRY.entity_types.by_key(entry.entity_type) else {
                    break;
                };
                max = entry.min_count
                    + random.next_i32_bounded(1 + entry.max_count - entry.min_count);
                current = Some((entry, entity_type));
            }

            let Some((entry, entity_type)) = current else {
                break;
            };

            if !is_valid_spawn_position_for_type(
                world,
                category,
                entry,
                entity_type,
                pos,
                nearest_player_distance_sqr,
                random,
            ) || !state.can_spawn(world, entity_type, pos)
            {
                continue;
            }

            let Some(entity) = get_mob_for_spawn(world, entity_type) else {
                return;
            };

            if !snap_to(&entity, center, random.next_f32() * 360.0, 0.0) {
                continue;
            }
            let Some(mob) = entity.as_mob() else {
                return;
            };

            if !is_valid_position_for_mob(world, mob, nearest_player_distance_sqr) {
                continue;
            }

            group_data = mob.finalize_spawn(world, EntitySpawnReason::Natural, group_data);
            cluster_size += 1;
            group_size += 1;
            let max_cluster_size = mob.get_max_spawn_cluster_size();
            let group_size_reached = mob.is_max_group_size_reached(group_size);

            if let Err(error) = world.try_add_entity(Arc::clone(&entity)) {
                log::warn!("Failed to add naturally spawned mob: {error:?}");
                continue;
            }
            state.after_spawn(world, entity_type, pos);

            if cluster_size >= max_cluster_size {
                return;
            }
            if group_size_reached {
                break;
            }
        }
    }
}

/// Vanilla `NaturalSpawner.spawnCategoryForChunk`.
///
/// One position per category per chunk per tick. The floor guard is vanilla's, and it is
/// `min_y + 1` rather than `min_y` because the position tests read the block *below* the
/// candidate.
fn spawn_category_for_chunk(
    world: &Arc<World>,
    category: MobCategory,
    chunk_pos: ChunkPos,
    state: &mut SpawnState<'_>,
    random: &mut impl Random,
) {
    let start = get_random_pos_within(world, chunk_pos, random);
    if start.y() > world.get_min_y() {
        spawn_category_for_position(world, category, chunk_pos, start, state, random);
    }
}

/// Vanilla `NaturalSpawner.spawnForChunk`.
pub(crate) fn spawn_for_chunk(
    world: &Arc<World>,
    chunk_pos: ChunkPos,
    state: &mut SpawnState<'_>,
    spawning_categories: &[MobCategory],
    random: &mut impl Random,
) {
    for &category in spawning_categories {
        if state.can_spawn_for_category_local(category, chunk_pos) {
            spawn_category_for_chunk(world, category, chunk_pos, state, random);
        }
    }
}

/// Runs one world's natural-spawn tick, vanilla's `ServerChunkCache.tickChunks` spawn phase.
///
/// Vanilla builds the [`SpawnState`] unconditionally — the mob cap census runs even when the
/// gamerule is off, because `/data` and the debug screen read it — and only the category list is
/// gated. The `400`-tick stride on persistent categories is vanilla's, and it is why animals
/// keep appearing long after a chunk's first visit but far more slowly than monsters.
///
/// The chunk order is vanilla's shuffle: without it the first chunks in traversal order would
/// take every spawn the caps allow.
pub(crate) fn tick_natural_spawns(world: &Arc<World>, holders: Vec<Arc<ChunkHolder>>) {
    let players = world.chunk_map.spawning_players();
    let spawnable_chunk_count = players.natural_spawn_chunk_count();
    let mut state = create_state(
        world,
        u32::try_from(spawnable_chunk_count).unwrap_or(u32::MAX),
        &players,
    );

    if !world.get_game_rule(&SPAWN_MOBS) {
        return;
    }

    let spawn_enemies = world.difficulty() != Difficulty::Peaceful;
    let spawn_persistent = world.game_time() % PERSISTENT_SPAWN_INTERVAL == 0;
    let spawning_categories = filtered_spawning_categories(&state, spawn_enemies, spawn_persistent);
    if spawning_categories.is_empty() {
        return;
    }

    let mut random = LegacyRandom::from_seed(rand::random());
    let mut holders = holders;
    vanilla_shuffle(&mut holders, &mut random);

    for holder in holders {
        let chunk_pos = holder.get_pos();
        if holder.try_full_chunk().is_none() {
            continue;
        }
        if !can_spawn_entities_in_chunk(world, chunk_pos) {
            continue;
        }
        spawn_for_chunk(
            world,
            chunk_pos,
            &mut state,
            &spawning_categories,
            &mut random,
        );
    }
}

/// Vanilla `Util.shuffle`, a Fisher-Yates walk from the top using the level's own randomness.
fn vanilla_shuffle<T>(items: &mut [T], random: &mut impl Random) {
    for index in (1..items.len()).rev() {
        let swap = random.next_i32_bounded(i32::try_from(index + 1).unwrap_or(i32::MAX));
        items.swap(index, usize::try_from(swap).unwrap_or(0));
    }
}

/// Number of blocks a chunk-generation group walks per member, vanilla's `nextInt(5)`.
const CHUNK_GENERATION_WALK_SPREAD: i32 = 5;

/// Placement attempts vanilla allows each member of a chunk-generation group.
const CHUNK_GENERATION_PLACEMENT_ATTEMPTS: i32 = 4;

/// Vanilla `NaturalSpawner.getTopNonCollidingPos`.
///
/// The ceiling loop is what keeps nether animals out of the roof: it walks down through the
/// bedrock lid until it leaves solid blocks, then down again through the cavity, so the position
/// it returns is under the ceiling rather than on top of it.
fn get_top_non_colliding_pos(
    region: &WorldGenRegion<'_>,
    entity_type: EntityTypeRef,
    x: i32,
    z: i32,
) -> BlockPos {
    let heightmap_type = spawn_placements::heightmap_type(entity_type);
    let mut pos = BlockPos::new(x, region.height_at(heightmap_type, x, z), z);

    if region.dimension_type().has_ceiling {
        loop {
            pos = pos.below();
            if region.block_state(pos).is_air() {
                break;
            }
        }
        loop {
            pos = pos.below();
            if !region.block_state(pos).is_air() || pos.y() <= region.min_y() {
                break;
            }
        }
    }

    spawn_placements::placement_type(entity_type).adjust_spawn_position(region, pos)
}

/// Vanilla `NaturalSpawner.spawnMobsForChunkGeneration`, the reason a fresh world already has
/// animals in it before anybody walks into the chunk.
///
/// Creatures only, and driven by the biome's own `creature_spawn_probability` rather than by the
/// mob caps — the caps do not exist yet at generation time. The `while` is vanilla's: a biome with
/// a high probability rolls several independent groups from one chunk.
///
/// Two bounded deviations from vanilla, both forced by Steel's `Mob` trait taking a live
/// [`World`] where vanilla takes the level accessor, and both on the *mob*-level checks rather
/// than the type-level ones:
///
/// * `check_spawn_obstruction` reads the live world, where this chunk is not present yet, so it
///   sees air and answers permissively. The solid-block half of vanilla's guard is already
///   enforced above by [`has_block_collision_in_level`], which does read the region, so what is
///   lost is only the fluid and entity overlap test on a chunk that has neither yet.
/// * `finalize_spawn` likewise receives the live world. For the creature types this path can
///   reach it sets baby chance and group data and reads nothing positional, so the difference is
///   not observable.
///
/// The type-level rules — light, the block below, biome — go through
/// [`spawn_placements::check_spawn_rules`], which does take the region, so the checks that decide
/// *whether a cow may stand here* are read from the chunk being generated.
pub(crate) fn spawn_mobs_for_chunk_generation(
    region: &WorldGenRegion<'_>,
    biome: BiomeRef,
    chunk_pos: ChunkPos,
    random: &mut impl Random,
) {
    let mobs = biome_spawners(biome, MobCategory::Creature);
    if mobs.is_empty() {
        return;
    }

    let Some(world) = region.weak_world().upgrade() else {
        return;
    };
    if !world.get_game_rule(&SPAWN_MOBS) {
        return;
    }

    let xo = chunk_pos.0.x * 16;
    let zo = chunk_pos.0.y * 16;
    let mut placed_total = 0_u32;

    while random.next_f32() < biome.creature_spawn_probability {
        let Some(entry) = mobs.get_random(random) else {
            continue;
        };
        let Some(entity_type) = REGISTRY.entity_types.by_key(entry.entity_type) else {
            continue;
        };

        let count =
            entry.min_count + random.next_i32_bounded(1 + entry.max_count - entry.min_count);
        let mut group_data = None;
        let start_x = xo + random.next_i32_bounded(16);
        let start_z = zo + random.next_i32_bounded(16);
        let mut x = start_x;
        let mut z = start_z;

        for _ in 0..count {
            let mut placed = false;
            let mut attempts = 0;

            while !placed && attempts < CHUNK_GENERATION_PLACEMENT_ATTEMPTS {
                attempts += 1;
                placed = try_place_generated_mob(
                    region,
                    &world,
                    entity_type,
                    (xo, zo),
                    (x, z),
                    &mut group_data,
                    random,
                );
                if placed {
                    placed_total += 1;
                }

                x += random.next_i32_bounded(CHUNK_GENERATION_WALK_SPREAD)
                    - random.next_i32_bounded(CHUNK_GENERATION_WALK_SPREAD);
                z += random.next_i32_bounded(CHUNK_GENERATION_WALK_SPREAD)
                    - random.next_i32_bounded(CHUNK_GENERATION_WALK_SPREAD);
                while x < xo || x >= xo + 16 || z < zo || z >= zo + 16 {
                    x = start_x + random.next_i32_bounded(CHUNK_GENERATION_WALK_SPREAD)
                        - random.next_i32_bounded(CHUNK_GENERATION_WALK_SPREAD);
                    z = start_z + random.next_i32_bounded(CHUNK_GENERATION_WALK_SPREAD)
                        - random.next_i32_bounded(CHUNK_GENERATION_WALK_SPREAD);
                }
            }
        }
    }

    if placed_total > 0 {
        log::debug!(
            "Chunk generation placed {placed_total} creatures in {chunk_pos:?} ({})",
            biome.key
        );
    }
}

/// Places one member of a chunk-generation group, vanilla's innermost attempt body.
///
/// Returns whether the mob was placed, which is vanilla's `success` flag and ends the attempt
/// loop for this member.
fn try_place_generated_mob(
    region: &WorldGenRegion<'_>,
    world: &Arc<World>,
    entity_type: EntityTypeRef,
    chunk_origin: (i32, i32),
    candidate: (i32, i32),
    group_data: &mut Option<SpawnGroupData>,
    random: &mut impl Random,
) -> bool {
    let (xo, zo) = chunk_origin;
    let (x, z) = candidate;

    let pos = get_top_non_colliding_pos(region, entity_type, x, z);
    if !entity_type.summonable || !spawn_placements::is_spawn_position_ok(entity_type, region, pos)
    {
        return false;
    }

    // Vanilla clamps by the full width so the box stays inside the chunk being generated.
    let width = f64::from(entity_type.dimensions.width);
    let fx = f64::from(x).clamp(f64::from(xo) + width, f64::from(xo) + 16.0 - width);
    let fz = f64::from(z).clamp(f64::from(zo) + width, f64::from(zo) + 16.0 - width);
    let center = DVec3::new(fx, f64::from(pos.y()), fz);

    if has_block_collision_in_level(
        region,
        WorldAabb::entity_box(
            fx,
            f64::from(pos.y()),
            fz,
            f64::from(entity_type.dimensions.half_width()),
            f64::from(entity_type.dimensions.height),
        ),
    ) {
        return false;
    }

    if !spawn_placements::check_spawn_rules(
        entity_type,
        region,
        EntitySpawnReason::ChunkGeneration,
        BlockPos::containing(fx, f64::from(pos.y()), fz),
        random,
    ) {
        return false;
    }

    let Some(entity) = get_mob_for_spawn(world, entity_type) else {
        return false;
    };
    if !snap_to(&entity, center, random.next_f32() * 360.0, 0.0) {
        return false;
    }
    let Some(mob) = entity.as_mob() else {
        return false;
    };
    if !mob.check_spawn_rules(world, EntitySpawnReason::ChunkGeneration)
        || !mob.check_spawn_obstruction(world)
    {
        return false;
    }

    *group_data = mob.finalize_spawn(world, EntitySpawnReason::ChunkGeneration, group_data.take());
    region.add_fresh_entity(Arc::clone(&entity))
}

#[cfg(test)]
mod tests {
    use core::f32::consts::SQRT_2;
    use std::sync::Arc;

    use glam::DVec3;
    use steel_registry::structure::TerrainAdjustment;
    use steel_registry::{
        RegistryEntry as _, init_vanilla_registry, vanilla_biomes, vanilla_entities,
    };
    use steel_utils::BoundingBox;
    use steel_utils::random::RandomSplitter;
    use steel_utils::types::UpdateFlags;
    use steel_worldgen::structure::{StructurePiece, StructureStart};

    use super::{
        BiomeRef, BlockPos, ChunkPos, INSCRIBED_SQUARE_SPAWN_DISTANCE_CHUNK, Identifier,
        LegacyRandom, MobCategory, PLAYER_SPAWN_EXCLUSION_DISTANCE_SQUARED, REGISTRY, Random,
        RegistryExt as _, SPAWN_CANDIDATE_SQUARE_CHUNKS, SPAWN_DISTANCE_BLOCK_SQUARED,
        SPAWN_DISTANCE_CHUNK, SpawnerEntry, SpawnerTable, World, biome_spawners,
        can_spawn_entities_in_chunk, can_spawn_mob_at, create_state, filtered_spawning_categories,
        fortress_enemies, get_mob_for_spawn, get_random_pos_within, is_in_nether_fortress_bounds,
        is_right_distance_to_player_and_spawn_point, is_valid_position_for_mob, mobs_at,
        random_spawn_mob_at, snap_to, spawn_for_chunk, vanilla_blocks, vanilla_shuffle,
    };
    use crate::behavior::init_behaviors;
    use crate::bootstrap::init_globals_once;
    use crate::chunk::chunk_holder::ChunkHolder;
    use crate::chunk::status::ChunkStatus;
    use crate::entity::entities::SheepEntity;
    use crate::entity::{Entity as _, EntityOwnership, SharedEntity, next_entity_id};
    use crate::player::{Player, ResetReason};
    use crate::test_support::{TestPlayerBuilder, fresh_test_world, insert_ready_full_chunk};

    /// Vanilla's cap for [`MobCategory::Creature`], the smallest of the land categories.
    const CREATURE_CAP: i32 = 10;

    /// Soul sand valley's charge for every type it prices, from its biome JSON.
    const PRICED_CHARGE: f64 = 0.7;

    /// Soul sand valley's energy budget for every type it prices, from its biome JSON.
    const PRICED_ENERGY_BUDGET: f64 = 0.15;

    /// Tolerance for the exactly-representable quotients these tests construct.
    const TOLERANCE: f64 = 1e-12;

    /// `y` of the grass floor the end-to-end test lays, with open air above it.
    const GRASS_FLOOR_Y: i32 = 63;

    /// Seeds the end-to-end test is allowed to try before it calls the commit path broken.
    const END_TO_END_SEEDS: u64 = 64;

    /// Counts the live [`MobCategory::Creature`] entities the world can see.
    fn creature_count(world: &Arc<World>) -> usize {
        world
            .entity_manager()
            .get_accessible_entities()
            .iter()
            .filter(|entity| entity.entity_type().mob_category == MobCategory::Creature)
            .count()
    }

    /// Fills one chunk's `y` layer with grass blocks, the only block animals may spawn on.
    fn lay_grass_floor(world: &Arc<World>, pos: ChunkPos, y: i32) {
        let min_x = pos.0.x * 16;
        let min_z = pos.0.y * 16;
        for x in min_x..min_x + 16 {
            for z in min_z..min_z + 16 {
                assert!(
                    world.set_block(
                        BlockPos::new(x, y, z),
                        vanilla_blocks::GRASS_BLOCK.default_state(),
                        UpdateFlags::UPDATE_NONE,
                    ),
                    "the floor's chunk should be loaded"
                );
            }
        }
    }

    /// The center of a chunk's block column, where the player filter measures distance from.
    fn chunk_center(chunk_x: i32, chunk_z: i32) -> DVec3 {
        DVec3::new(
            f64::from(chunk_x) * 16.0 + 8.0,
            64.0,
            f64::from(chunk_z) * 16.0 + 8.0,
        )
    }

    /// Builds a player standing at `position` and joins it to `world`.
    fn joined_player(world: &Arc<World>, name: &'static str, position: DVec3) -> Arc<Player> {
        let player = TestPlayerBuilder::new(Arc::clone(world), name, next_entity_id()).build();
        assert!(
            player.try_set_position(position).is_ok(),
            "test player should be placed before joining"
        );
        assert!(world.add_player(Arc::clone(&player), ResetReason::InitialJoin));
        player
    }

    /// Adds a sheep at `position`, the one [`MobCategory::Creature`] Steel can construct today.
    fn added_sheep(world: &Arc<World>, position: DVec3) -> SharedEntity {
        let sheep = SheepEntity::new(
            &vanilla_entities::SHEEP,
            next_entity_id(),
            position,
            Arc::downgrade(world),
        );
        let shared: SharedEntity = Arc::new(sheep);
        world
            .try_add_entity(Arc::clone(&shared))
            .expect("the sheep's chunk should be loaded");
        shared
    }

    /// Repaints every biome cell of a loaded chunk, the way the generators fill them.
    ///
    /// Only soul sand valley and warped forest declare `spawn_costs` at all, so a test that wants
    /// the crowding field to charge anything has to move off the test world's plains.
    fn paint_biome(world: &Arc<World>, pos: ChunkPos, biome_id: u16) {
        let painted = world.chunk_map.with_full_chunk(pos, |chunk| {
            for section in &chunk.sections().sections {
                let mut guard = section.write();
                for local_quart_x in 0..4 {
                    for local_quart_y in 0..4 {
                        for local_quart_z in 0..4 {
                            guard
                                .biomes
                                .set(local_quart_x, local_quart_y, local_quart_z, biome_id);
                        }
                    }
                }
            }
        });

        assert!(
            painted.is_some(),
            "the chunk being painted should be loaded"
        );
    }

    /// Asserts `value` is `expected`, which every quotient below is exactly.
    fn assert_close(value: f64, expected: f64) {
        assert!(
            (value - expected).abs() < TOLERANCE,
            "expected {expected}, got {value}"
        );
    }

    /// A [`Random`] that hands out a written-down sequence and refuses anything unscripted.
    ///
    /// The tables below are read for two things — one bounded draw to pick a row and one float to
    /// thin river fish — and *when* each is drawn is as much a part of the port as which row comes
    /// back. Panicking on an unexpected draw makes the draw order an assertion rather than a comment:
    /// a guard reordered so the float is consumed before the category is checked fails here, where a
    /// seeded generator would happily produce a plausible answer.
    struct ScriptedRandom {
        /// Values [`Random::next_i32_bounded`] returns, front first.
        bounded: Vec<i32>,
        /// Values [`Random::next_f32`] returns, front first.
        floats: Vec<f32>,
    }

    impl ScriptedRandom {
        /// A generator scripted for one bounded draw and no floats.
        fn bounded(value: i32) -> Self {
            Self {
                bounded: vec![value],
                floats: Vec::new(),
            }
        }

        /// A generator that panics the moment anything asks it for a number.
        fn silent() -> Self {
            Self {
                bounded: Vec::new(),
                floats: Vec::new(),
            }
        }

        /// Asserts the whole script was consumed, so an unused value cannot pass unnoticed.
        fn assert_exhausted(&self) {
            assert!(
                self.bounded.is_empty() && self.floats.is_empty(),
                "the scripted draws should all have been consumed"
            );
        }
    }

    impl Random for ScriptedRandom {
        fn fork(&mut self) -> Self {
            unreachable!("the spawn tables never fork")
        }

        fn next_i32(&mut self) -> i32 {
            unreachable!("the spawn tables only draw bounded integers")
        }

        fn next_i32_bounded(&mut self, bound: i32) -> i32 {
            assert!(
                !self.bounded.is_empty(),
                "unscripted draw bounded by {bound}"
            );
            let value = self.bounded.remove(0);
            assert!(
                value < bound,
                "scripted draw {value} is outside its bound {bound}"
            );
            value
        }

        fn next_i64(&mut self) -> i64 {
            unreachable!("the spawn tables never draw a long")
        }

        fn next_f32(&mut self) -> f32 {
            assert!(!self.floats.is_empty(), "unscripted float draw");
            self.floats.remove(0)
        }

        fn next_f64(&mut self) -> f64 {
            unreachable!("the spawn tables never draw a double")
        }

        fn next_bool(&mut self) -> bool {
            unreachable!("the spawn tables never draw a boolean")
        }

        fn next_gaussian(&mut self) -> f64 {
            unreachable!("the spawn tables never draw a gaussian")
        }

        fn next_positional(&mut self) -> RandomSplitter {
            unreachable!("the spawn tables never split")
        }
    }

    /// A biome by key, for the tables these tests read out of real vanilla data.
    fn biome(key: &Identifier) -> BiomeRef {
        REGISTRY
            .biomes
            .by_key(key)
            .expect("the vanilla registry should hold this biome")
    }

    /// Paints `pos` with `key`'s biome and answers the id, for asserting the read came back.
    fn paint(world: &Arc<World>, pos: ChunkPos, key: &Identifier) -> BiomeRef {
        let painted = biome(key);
        paint_biome(
            world,
            pos,
            u16::try_from(painted.id()).expect("a biome id should fit a u16"),
        );
        painted
    }

    /// A piece box spanning `min..=max`, with the metadata `non_jigsaw` gives a hand-built piece.
    fn piece(min: BlockPos, max: BlockPos) -> StructurePiece {
        StructurePiece::non_jigsaw(
            Identifier::new_static("steel", "test_piece"),
            BoundingBox::from_corners(min, max),
            0,
            None,
        )
    }

    /// Vanilla `setStartForStructure` plus `addReferenceForStructure`, for one self-referencing chunk.
    ///
    /// The reads only ever find a start by following a reference, so both halves are always written
    /// together. [`TerrainAdjustment::None`] keeps the inflate at zero, so the start's box is exactly
    /// the union of the pieces handed in.
    fn install_structure(
        holder: &ChunkHolder,
        structure: &Identifier,
        pieces: Vec<StructurePiece>,
    ) {
        let chunk = holder
            .try_chunk(ChunkStatus::Full)
            .expect("the test chunk should be published at Full");
        let start = StructureStart::new(
            structure.clone(),
            chunk.pos,
            pieces,
            TerrainAdjustment::None,
        );

        chunk
            .structure_starts_mut()
            .insert(structure.clone(), start);
        chunk
            .structure_references_mut()
            .entry(structure.clone())
            .or_default()
            .insert(chunk.pos);
    }

    /// The rows of a table as `(id, weight, min, max)`, which is how the JSON reads.
    fn rows_of(table: SpawnerTable) -> Vec<(String, i32, i32, i32)> {
        table
            .rows()
            .map(|row| {
                (
                    row.entity_type.to_string(),
                    row.weight,
                    row.min_count,
                    row.max_count,
                )
            })
            .collect()
    }

    /// The one row a table holds, for the overrides that declare exactly one.
    fn only_row(table: SpawnerTable) -> SpawnerEntry {
        let mut rows = table.rows();
        let row = rows.next().expect("the table should hold a row");
        assert!(rows.next().is_none(), "the table should hold only one row");
        row
    }

    /// The squared radius is the literal vanilla's per-player filter compares against.
    #[test]
    fn spawn_distance_block_squared_is_vanillas_literal() {
        assert_eq!(SPAWN_DISTANCE_BLOCK_SQUARED, 16_384);
    }

    /// The inscribed radius is the floor of vanilla's `8.0F / Mth.SQRT_OF_TWO`.
    ///
    /// Bracketing rather than comparing floats keeps the assertion exact.
    #[test]
    fn inscribed_square_radius_is_floor_of_vanillas_expression() {
        let radius = f64::from(SPAWN_DISTANCE_CHUNK) / f64::from(SQRT_2);
        assert!(radius >= f64::from(INSCRIBED_SQUARE_SPAWN_DISTANCE_CHUNK));
        assert!(radius < f64::from(INSCRIBED_SQUARE_SPAWN_DISTANCE_CHUNK + 1));
    }

    /// The inscribed radius is the largest one the block distance test can be skipped for.
    ///
    /// Worst case is a player standing on their own chunk's corner with the candidate chunk's
    /// center `radius` chunks away on both axes, so each axis differs by `radius * 16 + 8`.
    #[test]
    fn inscribed_square_radius_is_the_largest_sound_shortcut() {
        let corner_offset = |chunks: i64| chunks * 16 + 8;
        let inscribed = i64::from(INSCRIBED_SQUARE_SPAWN_DISTANCE_CHUNK);
        let bound = i64::from(SPAWN_DISTANCE_BLOCK_SQUARED);

        assert!(2 * corner_offset(inscribed).pow(2) < bound);
        assert!(2 * corner_offset(inscribed + 1).pow(2) >= bound);
    }

    /// The global cap's divisor is vanilla's `MAGIC_NUMBER`, `(int)Math.pow(17.0, 2.0)`.
    #[test]
    fn the_spawn_candidate_square_is_vanillas_magic_number() {
        assert_eq!(SPAWN_CANDIDATE_SQUARE_CHUNKS, 289);
    }

    /// The census counts a live mob and steps over the player, whose category is `Misc`.
    #[test]
    fn the_census_counts_a_mob_and_ignores_the_player() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_state_census");
        insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
        let player = joined_player(&world, "CensusWatcher", chunk_center(0, 0));
        added_sheep(&world, chunk_center(0, 0));

        // Both have to be visible to the walk, or the `Misc` skip below proves nothing.
        let accessible = world.entity_manager().get_accessible_entities();
        assert_eq!(
            accessible.len(),
            2,
            "the player and the sheep should both be accessible"
        );

        let players = world.chunk_map.spawning_players();
        let state = create_state(&world, SPAWN_CANDIDATE_SQUARE_CHUNKS, &players);

        assert_eq!(state.spawnable_chunk_count(), SPAWN_CANDIDATE_SQUARE_CHUNKS);
        assert_eq!(
            state.mob_category_counts().get(&MobCategory::Creature),
            Some(&1)
        );
        assert!(!state.mob_category_counts().contains_key(&MobCategory::Misc));

        world.remove_player_for_world_change(&player);
    }

    /// A mob the player made permanent is skipped, so it cannot suppress natural spawning.
    #[test]
    fn the_census_skips_a_mob_the_player_made_permanent() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_state_persistent");
        insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
        let sheep = added_sheep(&world, chunk_center(0, 0));
        sheep
            .as_mob()
            .expect("a sheep is a mob")
            .set_persistence_required();

        let players = world.chunk_map.spawning_players();
        let state = create_state(&world, SPAWN_CANDIDATE_SQUARE_CHUNKS, &players);

        assert!(
            !state
                .mob_category_counts()
                .contains_key(&MobCategory::Creature)
        );
    }

    /// An entity whose chunk is not loaded is not counted, matching vanilla's `ChunkGetter.query`.
    ///
    /// `External` ownership is what makes this observable: it keeps the entity accessible to
    /// lookups with no chunk behind it, which is exactly the state vanilla's query guards against.
    #[test]
    fn the_census_skips_an_entity_whose_chunk_is_not_loaded() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_state_unloaded");
        let sheep = SheepEntity::new(
            &vanilla_entities::SHEEP,
            next_entity_id(),
            chunk_center(40, 40),
            Arc::downgrade(&world),
        );
        let shared: SharedEntity = Arc::new(sheep);
        assert!(
            world
                .entity_manager()
                .add_live_entity(Arc::clone(&shared), EntityOwnership::External)
                .is_ok()
        );
        assert!(
            world
                .entity_manager()
                .get_accessible_entities()
                .iter()
                .any(|entity| entity.id() == shared.id()),
            "the far sheep should be accessible, or the skip under test is vacuous"
        );

        let players = world.chunk_map.spawning_players();
        let state = create_state(&world, SPAWN_CANDIDATE_SQUARE_CHUNKS, &players);

        assert!(
            !state
                .mob_category_counts()
                .contains_key(&MobCategory::Creature)
        );
    }

    /// The global allowance is the category's per-chunk figure scaled by the loaded area.
    #[test]
    fn the_global_cap_scales_the_per_chunk_allowance_by_loaded_area() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_state_global_cap");
        insert_ready_full_chunk(&world, ChunkPos::new(0, 0));

        let players = world.chunk_map.spawning_players();
        let mut state = create_state(&world, SPAWN_CANDIDATE_SQUARE_CHUNKS, &players);
        let pos = BlockPos::new(8, 64, 8);

        for _ in 0..CREATURE_CAP - 1 {
            state.after_spawn(&world, &vanilla_entities::SHEEP, pos);
        }
        assert!(state.can_spawn_for_category_global(MobCategory::Creature));

        state.after_spawn(&world, &vanilla_entities::SHEEP, pos);
        assert!(!state.can_spawn_for_category_global(MobCategory::Creature));
        // One player's worth of area allows exactly the per-chunk figure, and the tallies are
        // independent: filling one category leaves the others untouched.
        assert!(state.can_spawn_for_category_global(MobCategory::Monster));
    }

    /// The allowance is integer arithmetic, so a partial player's area rounds down.
    #[test]
    fn the_global_cap_rounds_a_partial_players_area_down() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_state_rounding");
        insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
        let players = world.chunk_map.spawning_players();
        let pos = BlockPos::new(8, 64, 8);

        // 10 * 288 / 289 is 9, one short of the full square's allowance.
        let mut state = create_state(&world, SPAWN_CANDIDATE_SQUARE_CHUNKS - 1, &players);
        for _ in 0..CREATURE_CAP - 1 {
            state.after_spawn(&world, &vanilla_entities::SHEEP, pos);
        }
        assert!(!state.can_spawn_for_category_global(MobCategory::Creature));

        // 10 * 28 / 289 is 0, so a world with almost nothing loaded spawns nothing at all.
        let barely_loaded = create_state(&world, 28, &players);
        assert!(!barely_loaded.can_spawn_for_category_global(MobCategory::Creature));
    }

    /// With both gamerules on and nothing alive, every category but `Misc`, in vanilla's order.
    #[test]
    fn the_category_filter_lists_every_spawning_category() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_filter_all");
        let players = world.chunk_map.spawning_players();
        let state = create_state(&world, SPAWN_CANDIDATE_SQUARE_CHUNKS, &players);

        assert_eq!(
            filtered_spawning_categories(&state, true, true),
            vec![
                MobCategory::Monster,
                MobCategory::Creature,
                MobCategory::Ambient,
                MobCategory::Axolotls,
                MobCategory::UndergroundWaterCreature,
                MobCategory::WaterCreature,
                MobCategory::WaterAmbient,
            ]
        );
    }

    /// Each flag drops exactly the categories vanilla ties to it, and `Misc` never appears.
    #[test]
    fn the_category_filter_honors_both_gamerule_flags() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_filter_flags");
        let players = world.chunk_map.spawning_players();
        let state = create_state(&world, SPAWN_CANDIDATE_SQUARE_CHUNKS, &players);

        let without_enemies = filtered_spawning_categories(&state, false, true);
        assert!(!without_enemies.contains(&MobCategory::Monster));
        assert!(without_enemies.contains(&MobCategory::Creature));

        let without_persistent = filtered_spawning_categories(&state, true, false);
        assert!(without_persistent.contains(&MobCategory::Monster));
        assert!(!without_persistent.contains(&MobCategory::Creature));

        let neither = filtered_spawning_categories(&state, false, false);
        for categories in [&without_enemies, &without_persistent, &neither] {
            assert!(!categories.contains(&MobCategory::Misc));
        }
    }

    /// A category already at its global allowance drops out of the filter on its own.
    #[test]
    fn a_capped_category_is_dropped_from_the_filter() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_filter_capped");
        insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
        let players = world.chunk_map.spawning_players();
        let mut state = create_state(&world, SPAWN_CANDIDATE_SQUARE_CHUNKS, &players);

        for _ in 0..CREATURE_CAP {
            state.after_spawn(&world, &vanilla_entities::SHEEP, BlockPos::new(8, 64, 8));
        }

        let categories = filtered_spawning_categories(&state, true, true);
        assert!(!categories.contains(&MobCategory::Creature));
        assert!(categories.contains(&MobCategory::Monster));
    }

    /// A type its biome prices nothing for passes however crowded the block already is.
    #[test]
    fn an_unpriced_type_always_passes_the_crowding_check() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_state_unpriced");
        let chunk = ChunkPos::new(0, 0);
        insert_ready_full_chunk(&world, chunk);
        paint_biome(&world, chunk, vanilla_biomes::SOUL_SAND_VALLEY.id() as u16);

        let players = world.chunk_map.spawning_players();
        let mut state = create_state(&world, SPAWN_CANDIDATE_SQUARE_CHUNKS, &players);
        let pos = BlockPos::new(8, 64, 8);
        state.after_spawn(&world, &vanilla_entities::SKELETON, pos);

        // The same block is infinitely expensive for the priced type and free for the unpriced one.
        assert!(!state.can_spawn(&world, &vanilla_entities::SKELETON, pos));
        assert!(state.can_spawn(&world, &vanilla_entities::SHEEP, pos));
    }

    /// The energy budget is a distance test in disguise: three blocks is too close, four is not.
    #[test]
    fn a_priced_type_is_refused_only_while_a_charge_is_too_close() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_state_priced");
        let chunk = ChunkPos::new(0, 0);
        insert_ready_full_chunk(&world, chunk);
        paint_biome(&world, chunk, vanilla_biomes::SOUL_SAND_VALLEY.id() as u16);

        let players = world.chunk_map.spawning_players();
        let mut state = create_state(&world, SPAWN_CANDIDATE_SQUARE_CHUNKS, &players);
        state.after_spawn(&world, &vanilla_entities::SKELETON, BlockPos::new(8, 64, 8));

        // 0.7 * 0.7 over three blocks exceeds the 0.15 budget; over four it does not.
        const { assert!(PRICED_CHARGE * PRICED_CHARGE / 3.0 > PRICED_ENERGY_BUDGET) }
        const { assert!(PRICED_CHARGE * PRICED_CHARGE / 4.0 <= PRICED_ENERGY_BUDGET) }
        assert!(!state.can_spawn(
            &world,
            &vanilla_entities::SKELETON,
            BlockPos::new(11, 64, 8)
        ));
        assert!(state.can_spawn(
            &world,
            &vanilla_entities::SKELETON,
            BlockPos::new(12, 64, 8)
        ));
    }

    /// A matching query and spawn reuse the memo, charging the field the biome's own figure.
    #[test]
    fn the_memo_charges_the_field_when_the_query_matches() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_state_memo_hit");
        let chunk = ChunkPos::new(0, 0);
        insert_ready_full_chunk(&world, chunk);
        paint_biome(&world, chunk, vanilla_biomes::SOUL_SAND_VALLEY.id() as u16);

        let players = world.chunk_map.spawning_players();
        let mut state = create_state(&world, SPAWN_CANDIDATE_SQUARE_CHUNKS, &players);
        let pos = BlockPos::new(8, 64, 8);

        assert!(state.can_spawn(&world, &vanilla_entities::SKELETON, pos));
        state.after_spawn(&world, &vanilla_entities::SKELETON, pos);

        // 0.7 * 0.7 read from seven blocks away.
        assert_close(
            state
                .spawn_potential
                .potential_energy_change(BlockPos::new(1, 64, 8), PRICED_CHARGE),
            PRICED_CHARGE * PRICED_CHARGE / 7.0,
        );
    }

    /// The memo is dropped when the caller moves the mob between the two calls.
    #[test]
    fn the_cost_memo_is_dropped_when_the_position_moves() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_state_memo_moved");
        let priced_chunk = ChunkPos::new(0, 0);
        insert_ready_full_chunk(&world, priced_chunk);
        insert_ready_full_chunk(&world, ChunkPos::new(1, 0));
        paint_biome(
            &world,
            priced_chunk,
            vanilla_biomes::SOUL_SAND_VALLEY.id() as u16,
        );

        let players = world.chunk_map.spawning_players();
        let mut state = create_state(&world, SPAWN_CANDIDATE_SQUARE_CHUNKS, &players);
        let priced = BlockPos::new(8, 64, 8);
        let plains = BlockPos::new(24, 64, 8);

        assert!(state.can_spawn(&world, &vanilla_entities::SKELETON, priced));
        state.after_spawn(&world, &vanilla_entities::SKELETON, plains);

        // Plains prices nothing, so a correct re-read charges zero. A stale 0.7 would have made
        // this block infinitely expensive instead.
        assert_eq!(
            state
                .spawn_potential
                .potential_energy_change(plains, PRICED_CHARGE),
            0.0
        );
    }

    /// The memo is dropped when the type changes, even at the very same block.
    #[test]
    fn the_cost_memo_is_dropped_when_the_type_changes() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_state_memo_retyped");
        let chunk = ChunkPos::new(0, 0);
        insert_ready_full_chunk(&world, chunk);
        paint_biome(&world, chunk, vanilla_biomes::SOUL_SAND_VALLEY.id() as u16);

        let players = world.chunk_map.spawning_players();
        let mut state = create_state(&world, SPAWN_CANDIDATE_SQUARE_CHUNKS, &players);
        let pos = BlockPos::new(8, 64, 8);

        assert!(state.can_spawn(&world, &vanilla_entities::SKELETON, pos));
        state.after_spawn(&world, &vanilla_entities::SHEEP, pos);

        // Soul sand valley prices skeletons but not sheep, so the re-read charges zero.
        assert_eq!(
            state
                .spawn_potential
                .potential_energy_change(pos, PRICED_CHARGE),
            0.0
        );
    }

    /// `after_spawn` books the mob into the per-player cap as well as the global one.
    #[test]
    fn after_spawn_books_the_mob_into_the_local_cap() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_state_local_cap");
        let chunk = ChunkPos::new(0, 0);
        insert_ready_full_chunk(&world, chunk);
        let player = joined_player(&world, "LocalCapBooked", chunk_center(0, 0));

        let players = world.chunk_map.spawning_players();
        let mut state = create_state(&world, SPAWN_CANDIDATE_SQUARE_CHUNKS, &players);
        assert!(state.can_spawn_for_category_local(MobCategory::Creature, chunk));

        for _ in 0..CREATURE_CAP {
            state.after_spawn(&world, &vanilla_entities::SHEEP, BlockPos::new(8, 64, 8));
        }

        assert!(!state.can_spawn_for_category_local(MobCategory::Creature, chunk));

        world.remove_player_for_world_change(&player);
    }

    /// A biome's table is its own category's list, read straight out of the generated JSON.
    ///
    /// Plains' six `creature` rows in declaration order, which is also weight order here by
    /// coincidence rather than by rule — the walk in [`SpawnerTable::get_random`] does not care.
    #[test]
    fn a_biomes_table_is_its_categorys_own_list() {
        init_vanilla_registry();
        let plains = biome(&vanilla_biomes::PLAINS.key);

        let creature = biome_spawners(plains, MobCategory::Creature);

        assert_eq!(creature.len(), 6);
        assert!(!creature.is_empty());
        assert_eq!(
            rows_of(creature),
            vec![
                ("minecraft:sheep".to_owned(), 12, 4, 4),
                ("minecraft:pig".to_owned(), 10, 4, 4),
                ("minecraft:chicken".to_owned(), 10, 4, 4),
                ("minecraft:cow".to_owned(), 8, 4, 4),
                ("minecraft:horse".to_owned(), 5, 2, 6),
                ("minecraft:donkey".to_owned(), 1, 1, 3),
            ]
        );
        assert_eq!(creature.total_weight(), 46);
    }

    /// A category the biome does not declare is vanilla's `EMPTY_MOB_LIST`.
    ///
    /// Plains declares four of the eight categories, and the void declares none at all — the two
    /// ways an absent key is reached.
    #[test]
    fn an_undeclared_category_reads_empty() {
        init_vanilla_registry();
        let plains = biome(&vanilla_biomes::PLAINS.key);
        let void = biome(&vanilla_biomes::THE_VOID.key);

        let absent = biome_spawners(plains, MobCategory::Axolotls);
        assert!(absent.is_empty());
        assert_eq!(absent.total_weight(), 0);
        assert_eq!(rows_of(absent), []);

        assert!(biome_spawners(void, MobCategory::Creature).is_empty());
        assert!(biome_spawners(void, MobCategory::Monster).is_empty());
    }

    /// The fortress list comes out of the structure registry, matching vanilla's hardcoded one.
    ///
    /// Vanilla builds `FORTRESS_ENEMIES` in Java *and* ships the identical list as fortress's
    /// `spawn_overrides.monster`; this asserts the shipped copy row for row, so a regression in the
    /// structure loader shows up here rather than as a nether with no blazes in it.
    #[test]
    fn the_fortress_list_is_read_from_the_registry() {
        init_vanilla_registry();

        assert_eq!(
            rows_of(fortress_enemies(MobCategory::Monster)),
            vec![
                ("minecraft:blaze".to_owned(), 10, 2, 3),
                ("minecraft:zombified_piglin".to_owned(), 5, 4, 4),
                ("minecraft:wither_skeleton".to_owned(), 8, 5, 5),
                ("minecraft:skeleton".to_owned(), 2, 5, 5),
                ("minecraft:magma_cube".to_owned(), 3, 4, 4),
            ]
        );
        assert_eq!(fortress_enemies(MobCategory::Monster).total_weight(), 28);
        // Fortress overrides only `monster`, so every other category answers empty.
        assert!(fortress_enemies(MobCategory::Creature).is_empty());
    }

    /// The walk subtracts each row's weight in declaration order, boundaries included.
    ///
    /// Plains' `creature` weights are 12, 10, 10, 8, 5, 1 over a total of 46, so the last draw each
    /// row owns is 11, 21, 31, 39, 44, 45. Testing the last of each range is what catches an
    /// off-by-one in `index -= weight; if index < 0`.
    #[test]
    fn the_walk_hands_each_row_its_own_slice_of_the_total() {
        init_vanilla_registry();
        let creature = biome_spawners(biome(&vanilla_biomes::PLAINS.key), MobCategory::Creature);

        for (draw, expected) in [
            (0, "minecraft:sheep"),
            (11, "minecraft:sheep"),
            (12, "minecraft:pig"),
            (21, "minecraft:pig"),
            (22, "minecraft:chicken"),
            (31, "minecraft:chicken"),
            (32, "minecraft:cow"),
            (39, "minecraft:cow"),
            (40, "minecraft:horse"),
            (44, "minecraft:horse"),
            (45, "minecraft:donkey"),
        ] {
            let mut random = ScriptedRandom::bounded(draw);
            let row = creature
                .get_random(&mut random)
                .expect("a table with weight should always draw a row");

            assert_eq!(row.entity_type.to_string(), expected, "draw {draw}");
            random.assert_exhausted();
        }
    }

    /// An empty table draws nothing and consumes no randomness doing it.
    ///
    /// Vanilla's selector is null for a zero total, so `getRandom` returns before it would have
    /// called `nextInt` — and a `nextInt(0)` would throw if it did not. The silent generator makes
    /// the absent draw the assertion.
    #[test]
    fn an_empty_table_draws_nothing_and_asks_for_nothing() {
        init_vanilla_registry();
        let mut random = ScriptedRandom::silent();

        assert!(SpawnerTable::EMPTY.get_random(&mut random).is_none());
        assert!(SpawnerTable::EMPTY.is_empty());
        assert_eq!(SpawnerTable::EMPTY.len(), 0);
        assert_eq!(SpawnerTable::EMPTY.total_weight(), 0);
    }

    /// Membership ignores the weight, because vanilla's does.
    ///
    /// `WeightedList.contains` compares the value, and the weight lives on the wrapper — so the same
    /// mob with the same count range matches at any weight, while a different count range does not.
    #[test]
    fn membership_ignores_the_weight_but_not_the_counts() {
        init_vanilla_registry();
        let creature = biome_spawners(biome(&vanilla_biomes::PLAINS.key), MobCategory::Creature);
        let sheep = creature
            .rows()
            .next()
            .expect("plains should offer a first creature row");

        assert!(creature.contains(sheep));
        assert!(creature.contains(SpawnerEntry {
            weight: sheep.weight + 100,
            ..sheep
        }));
        assert!(!creature.contains(SpawnerEntry {
            min_count: sheep.min_count + 1,
            ..sheep
        }));
        assert!(!creature.contains(SpawnerEntry {
            max_count: sheep.max_count + 1,
            ..sheep
        }));
        // A mob from another table entirely, to show membership is not merely "some row matched".
        let blaze = fortress_enemies(MobCategory::Monster)
            .rows()
            .next()
            .expect("fortress should offer a first monster row");
        assert!(!creature.contains(blaze));
    }

    /// With no structure in reach, the table is the biome's.
    ///
    /// Also pins the parameter: handing in the caller's already-fuzzed read and letting [`mobs_at`]
    /// re-read it must give the same table, which is what makes threading the biome through an
    /// optimization rather than a behaviour change.
    #[test]
    fn a_position_outside_every_structure_reads_its_biome() {
        init_vanilla_registry();
        let world = fresh_test_world("natural_spawner_biome_table");
        let chunk = ChunkPos::new(0, 0);
        insert_ready_full_chunk(&world, chunk);
        let plains = paint(&world, chunk, &vanilla_biomes::PLAINS.key);
        let pos = BlockPos::new(8, 64, 8);

        let expected = rows_of(biome_spawners(plains, MobCategory::Creature));

        assert_eq!(
            rows_of(mobs_at(&world, MobCategory::Creature, pos, None)),
            expected
        );
        assert_eq!(
            rows_of(mobs_at(&world, MobCategory::Creature, pos, Some(plains))),
            expected
        );
    }

    /// A position outside every loaded chunk reads the generator's biome, generating nothing.
    ///
    /// Vanilla's `getNoiseBiome` asks for the chunk with `load = false` and falls back to
    /// `getUncachedNoiseBiome`, which samples the generator's biome source; Steel's `noise_biome_id`
    /// has the same fallback, so this is faithful rather than a shortcut. The test world's generator
    /// answers plains everywhere, so the unloaded read is plains' own table.
    #[test]
    fn an_unloaded_position_reads_the_generators_biome() {
        init_vanilla_registry();
        let world = fresh_test_world("natural_spawner_unloaded_table");
        let plains = biome(&vanilla_biomes::PLAINS.key);

        assert_eq!(
            rows_of(mobs_at(
                &world,
                MobCategory::Creature,
                BlockPos::new(8, 64, 8),
                None
            )),
            rows_of(biome_spawners(plains, MobCategory::Creature))
        );
    }

    /// An override with no rows suppresses the biome instead of falling through to it.
    ///
    /// This is the whole point of the empty list: ancient city declares all eight categories with no
    /// spawns and a `full` box, and trial chambers do the same with a `piece` box. Both must silence
    /// plains' nine monster rows.
    #[test]
    fn an_empty_override_suppresses_the_biome() {
        init_vanilla_registry();
        let world = fresh_test_world("natural_spawner_suppression");
        let chunk_pos = ChunkPos::new(0, 0);
        let chunk = insert_ready_full_chunk(&world, chunk_pos);
        paint(&world, chunk_pos, &vanilla_biomes::PLAINS.key);
        let pos = BlockPos::new(8, 64, 8);

        // The biome does offer monsters here, so an empty answer below can only come from the
        // override.
        assert!(!mobs_at(&world, MobCategory::Monster, pos, None).is_empty());

        install_structure(
            &chunk,
            &Identifier::vanilla_static("ancient_city"),
            vec![piece(BlockPos::new(0, 60, 0), BlockPos::new(15, 70, 15))],
        );

        assert!(mobs_at(&world, MobCategory::Monster, pos, None).is_empty());
        assert!(mobs_at(&world, MobCategory::Creature, pos, None).is_empty());
    }

    /// A `piece` override claims its pieces and leaves the gaps to the biome.
    ///
    /// Swamp hut's witch is the canonical case — the hut spawns witches, the swamp around it does
    /// not. Two pieces with a gap between them separate the two answers at one position each.
    #[test]
    fn a_piece_override_leaves_the_gaps_to_the_biome() {
        init_vanilla_registry();
        let world = fresh_test_world("natural_spawner_piece_override");
        let chunk_pos = ChunkPos::new(0, 0);
        let chunk = insert_ready_full_chunk(&world, chunk_pos);
        paint(&world, chunk_pos, &vanilla_biomes::PLAINS.key);
        install_structure(
            &chunk,
            &Identifier::vanilla_static("swamp_hut"),
            vec![
                piece(BlockPos::new(0, 60, 0), BlockPos::new(3, 70, 3)),
                piece(BlockPos::new(12, 60, 12), BlockPos::new(15, 70, 15)),
            ],
        );

        let inside = only_row(mobs_at(
            &world,
            MobCategory::Monster,
            BlockPos::new(2, 64, 2),
            None,
        ));
        assert_eq!(inside.entity_type.to_string(), "minecraft:witch");

        // The gap is inside the start's union but inside no piece, so the biome answers instead.
        // Plains offers witches too, so the test is that the whole biome list comes back rather than
        // the override's single row.
        let gap = rows_of(mobs_at(
            &world,
            MobCategory::Monster,
            BlockPos::new(8, 64, 8),
            None,
        ));
        assert_eq!(
            gap,
            rows_of(biome_spawners(
                biome(&vanilla_biomes::PLAINS.key),
                MobCategory::Monster
            ))
        );
    }

    /// The fortress branch tests the whole start's box, and needs nether bricks underfoot.
    ///
    /// The two fortress paths differ in exactly this: the generic scan reads fortress's `piece`
    /// override, while `isInNetherFortressBounds` accepts anywhere in the start's union that has
    /// nether bricks below it. A position in the gap between two pieces separates them — without the
    /// bricks the generic scan declines and the biome answers, with them the fortress list does.
    #[test]
    fn the_fortress_branch_claims_the_gaps_that_stand_on_nether_bricks() {
        init_vanilla_registry();
        let world = fresh_test_world("natural_spawner_fortress");
        let chunk_pos = ChunkPos::new(0, 0);
        let chunk = insert_ready_full_chunk(&world, chunk_pos);
        paint(&world, chunk_pos, &vanilla_biomes::PLAINS.key);
        install_structure(
            &chunk,
            &Identifier::vanilla_static("fortress"),
            vec![
                piece(BlockPos::new(0, 60, 0), BlockPos::new(3, 70, 3)),
                piece(BlockPos::new(12, 60, 12), BlockPos::new(15, 70, 15)),
            ],
        );
        let gap = BlockPos::new(8, 64, 8);

        assert!(!is_in_nether_fortress_bounds(
            &world,
            gap,
            MobCategory::Monster
        ));
        let without_bricks = rows_of(mobs_at(&world, MobCategory::Monster, gap, None));
        assert!(
            !without_bricks
                .iter()
                .any(|(id, ..)| id == "minecraft:blaze")
        );

        assert!(world.set_block(
            gap.below(),
            vanilla_blocks::NETHER_BRICKS.default_state(),
            UpdateFlags::UPDATE_NONE,
        ));

        assert!(is_in_nether_fortress_bounds(
            &world,
            gap,
            MobCategory::Monster
        ));
        assert_eq!(
            rows_of(mobs_at(&world, MobCategory::Monster, gap, None)),
            rows_of(fortress_enemies(MobCategory::Monster))
        );

        // The branch is monster-only, so the bricks change nothing for any other category.
        assert!(!is_in_nether_fortress_bounds(
            &world,
            gap,
            MobCategory::Creature
        ));
    }

    /// The nether-brick test reads the block below, not the block at the position.
    #[test]
    fn the_fortress_branch_reads_the_block_below() {
        init_vanilla_registry();
        let world = fresh_test_world("natural_spawner_fortress_below");
        let chunk_pos = ChunkPos::new(0, 0);
        let chunk = insert_ready_full_chunk(&world, chunk_pos);
        install_structure(
            &chunk,
            &Identifier::vanilla_static("fortress"),
            vec![piece(BlockPos::new(0, 60, 0), BlockPos::new(15, 70, 15))],
        );
        let pos = BlockPos::new(8, 64, 8);

        assert!(world.set_block(
            pos,
            vanilla_blocks::NETHER_BRICKS.default_state(),
            UpdateFlags::UPDATE_NONE,
        ));
        assert!(!is_in_nether_fortress_bounds(
            &world,
            pos,
            MobCategory::Monster
        ));
    }

    /// Rivers throw away almost every water-ambient attempt; oceans throw away none.
    ///
    /// The guard order is the assertion here. An untagged biome must not draw the float at all, so
    /// the ocean case is scripted with one bounded draw and no floats — a port that drew the float
    /// first would panic instead of quietly desynchronising every later roll.
    #[test]
    fn rivers_thin_their_water_ambient_spawns() {
        init_vanilla_registry();
        let world = fresh_test_world("natural_spawner_river_thinning");
        let chunk_pos = ChunkPos::new(0, 0);
        insert_ready_full_chunk(&world, chunk_pos);
        let pos = BlockPos::new(8, 64, 8);

        paint(&world, chunk_pos, &vanilla_biomes::RIVER.key);
        let mut discarded = ScriptedRandom {
            bounded: Vec::new(),
            floats: vec![0.5],
        };
        assert!(
            random_spawn_mob_at(&world, MobCategory::WaterAmbient, pos, &mut discarded).is_none()
        );
        discarded.assert_exhausted();

        // At or above the chance the attempt survives, and the river's only row is salmon.
        let mut kept = ScriptedRandom {
            bounded: vec![0],
            floats: vec![0.99],
        };
        let salmon = random_spawn_mob_at(&world, MobCategory::WaterAmbient, pos, &mut kept)
            .expect("a surviving river attempt should draw the river's own row");
        assert_eq!(salmon.entity_type.to_string(), "minecraft:salmon");
        kept.assert_exhausted();

        // Ocean is not in the tag, so no float is drawn and its own row comes back.
        paint(&world, chunk_pos, &vanilla_biomes::OCEAN.key);
        let mut untagged = ScriptedRandom::bounded(0);
        let cod = random_spawn_mob_at(&world, MobCategory::WaterAmbient, pos, &mut untagged)
            .expect("an ocean attempt should draw the ocean's own row");
        assert_eq!(cod.entity_type.to_string(), "minecraft:cod");
        untagged.assert_exhausted();
    }

    /// The river clause is water-ambient only, so no other category draws the float.
    #[test]
    fn the_river_clause_is_water_ambient_only() {
        init_vanilla_registry();
        let world = fresh_test_world("natural_spawner_river_category");
        let chunk_pos = ChunkPos::new(0, 0);
        insert_ready_full_chunk(&world, chunk_pos);
        paint(&world, chunk_pos, &vanilla_biomes::RIVER.key);
        let pos = BlockPos::new(8, 64, 8);

        let mut random = ScriptedRandom::bounded(0);
        let row = random_spawn_mob_at(&world, MobCategory::WaterCreature, pos, &mut random)
            .expect("the river should offer a water creature");

        assert_eq!(row.entity_type.to_string(), "minecraft:squid");
        random.assert_exhausted();
    }

    /// The group's drifted position is re-asked, and a suppressed one refuses the row.
    ///
    /// Vanilla calls this per group member after moving the position up to five blocks, so the row
    /// drawn in a plains chunk has to be re-validated wherever the group wandered to.
    #[test]
    fn a_drawn_row_is_revalidated_at_the_position_it_lands_on() {
        init_vanilla_registry();
        let world = fresh_test_world("natural_spawner_revalidation");
        let open = ChunkPos::new(0, 0);
        let suppressed = ChunkPos::new(1, 0);
        insert_ready_full_chunk(&world, open);
        let suppressed_chunk = insert_ready_full_chunk(&world, suppressed);
        paint(&world, open, &vanilla_biomes::PLAINS.key);
        paint(&world, suppressed, &vanilla_biomes::PLAINS.key);
        install_structure(
            &suppressed_chunk,
            &Identifier::vanilla_static("ancient_city"),
            vec![piece(BlockPos::new(16, 60, 0), BlockPos::new(31, 70, 15))],
        );

        let inside_plains = BlockPos::new(8, 64, 8);
        let inside_city = BlockPos::new(24, 64, 8);
        let sheep = biome_spawners(biome(&vanilla_biomes::PLAINS.key), MobCategory::Creature)
            .rows()
            .next()
            .expect("plains should offer a first creature row");

        assert!(can_spawn_mob_at(
            &world,
            MobCategory::Creature,
            inside_plains,
            sheep
        ));
        assert!(!can_spawn_mob_at(
            &world,
            MobCategory::Creature,
            inside_city,
            sheep
        ));
    }

    /// The drawn position is chunk-local in `x`/`z` and column-bounded in `y`.
    ///
    /// An `insert_ready_full_chunk` chunk is all air, so its world-surface heightmap reads back the
    /// world floor and the `y` draw collapses to a single legal value. That is the degenerate end
    /// of the range and worth pinning: it is what a spawn attempt in an ungenerated column does,
    /// and the scripted bound of `1` proves the range was computed rather than guessed.
    #[test]
    fn a_drawn_position_stays_inside_the_chunk_and_its_column() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_loop_random_pos");
        let chunk = ChunkPos::new(2, -3);
        insert_ready_full_chunk(&world, chunk);

        let mut random = ScriptedRandom {
            bounded: vec![5, 9, 0],
            floats: Vec::new(),
        };
        let pos = get_random_pos_within(&world, chunk, &mut random);
        random.assert_exhausted();

        assert_eq!(
            pos,
            BlockPos::new(2 * 16 + 5, world.get_min_y(), -3 * 16 + 9)
        );
    }

    /// Vanilla's `576.0` is a `<=`, so a mob may not stand exactly twenty-four blocks from a player.
    ///
    /// The candidate sits far from the world's respawn point so that only the player term can
    /// decide, and `origin_chunk` is the candidate's own chunk so the eligibility term passes
    /// without a second loaded chunk.
    #[test]
    fn the_player_exclusion_radius_is_twenty_four_blocks() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_loop_player_distance");
        let far = BlockPos::new(4096, 64, 4096);
        let origin = ChunkPos::from_block_pos(far);

        assert!(!is_right_distance_to_player_and_spawn_point(
            &world,
            origin,
            far,
            PLAYER_SPAWN_EXCLUSION_DISTANCE_SQUARED
        ));
        assert!(is_right_distance_to_player_and_spawn_point(
            &world,
            origin,
            far,
            PLAYER_SPAWN_EXCLUSION_DISTANCE_SQUARED + 1.0
        ));
    }

    /// The respawn point carries its own exclusion zone, of the same radius as the player's.
    #[test]
    fn the_respawn_point_keeps_its_own_twenty_four_blocks_clear() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_loop_respawn_exclusion");
        let respawn = {
            let level_data = world.level_data.read();
            level_data.data().respawn_data_or_local(&world.key).pos()
        };
        let origin = ChunkPos::from_block_pos(respawn);

        // Far enough from every player, but standing on the respawn block itself.
        assert!(!is_right_distance_to_player_and_spawn_point(
            &world,
            origin,
            respawn,
            f64::MAX
        ));
    }

    /// A chunk nothing has loaded holds no entities, so vanilla will not spawn into it.
    #[test]
    fn an_unloaded_chunk_accepts_no_entities() {
        init_vanilla_registry();
        init_behaviors();
        let world = fresh_test_world("spawn_loop_chunk_eligibility");

        assert!(!can_spawn_entities_in_chunk(&world, ChunkPos::new(7, 7)));
    }

    /// The shuffle reorders the candidate chunks without losing or duplicating one.
    #[test]
    fn the_chunk_shuffle_is_a_permutation() {
        let mut items: Vec<i32> = (0..32).collect();
        let mut random = LegacyRandom::from_seed(0x5EED);

        vanilla_shuffle(&mut items, &mut random);

        let mut sorted = items.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..32).collect::<Vec<_>>());
        assert_ne!(items, sorted, "a 32-element shuffle should reorder");
    }

    /// The factory answers for a ported class and declines the rest, which is the loop's skip.
    ///
    /// Plains offers horses and donkeys alongside the four farm animals, and neither has an entity
    /// class yet. Vanilla's `getMobForSpawn` returns null for a type it cannot build and the caller
    /// abandons the group; this pins both halves so a later port cannot quietly change which side a
    /// type falls on.
    #[test]
    fn the_mob_factory_answers_for_ported_classes_only() {
        init_vanilla_registry();
        init_behaviors();
        init_globals_once();
        let world = fresh_test_world("spawn_loop_mob_factory");

        let sheep = get_mob_for_spawn(&world, &vanilla_entities::SHEEP)
            .expect("sheep is one of the ported entity classes");
        assert_eq!(sheep.entity_type(), &vanilla_entities::SHEEP);
        assert!(sheep.as_mob().is_some(), "a sheep should be a mob");

        assert!(
            get_mob_for_spawn(&world, &vanilla_entities::HORSE).is_none(),
            "horse has no entity class yet, so the loop must skip it"
        );
    }

    /// A built sheep in open air passes the mob-side tests at every distance a player can be at.
    ///
    /// The first clause is the one worth pinning: a sheep is an `Animal`, which vanilla exempts
    /// from distance despawning once it has been counted, so `f64::MAX` has to pass rather than
    /// fail.
    #[test]
    fn a_built_sheep_passes_the_mob_side_position_tests() {
        init_vanilla_registry();
        init_behaviors();
        init_globals_once();
        let world = fresh_test_world("spawn_loop_mob_position");
        let chunk = ChunkPos::new(0, 0);
        insert_ready_full_chunk(&world, chunk);

        let sheep = get_mob_for_spawn(&world, &vanilla_entities::SHEEP)
            .expect("sheep is one of the ported entity classes");
        snap_to(&sheep, DVec3::new(8.5, 64.0, 8.5), 90.0, 0.0);
        let mob = sheep.as_mob().expect("a sheep should be a mob");

        assert!(is_valid_position_for_mob(&world, mob, 4096.0));
        assert!(is_valid_position_for_mob(&world, mob, f64::MAX));
    }

    /// `snap_to` commits both halves of vanilla's snap: the position and the rotation.
    #[test]
    fn snapping_a_built_mob_sets_its_position_and_rotation() {
        init_vanilla_registry();
        init_behaviors();
        init_globals_once();
        let world = fresh_test_world("spawn_loop_snap");
        insert_ready_full_chunk(&world, ChunkPos::new(0, 0));

        let sheep = get_mob_for_spawn(&world, &vanilla_entities::SHEEP)
            .expect("sheep is one of the ported entity classes");
        let target = DVec3::new(8.5, 64.0, 9.5);
        snap_to(&sheep, target, 123.0, 0.0);

        assert_eq!(sheep.position(), target);
        assert_eq!(sheep.rotation().0.to_bits(), 123.0_f32.to_bits());
        assert_eq!(sheep.base().old_position(), target);
    }

    /// The whole loop, end to end: a plains chunk with a grass floor produces live creatures.
    ///
    /// This is the test the rest of the module cannot replace. Every piece above is checked in
    /// isolation, and a spawn system whose pieces all pass while nothing ever appears in the world
    /// is exactly the failure this guards against — so the assertion is on the world's entity count
    /// rather than on any predicate's return value.
    ///
    /// The loop is probabilistic by construction, so the test drives a run of seeds and requires
    /// that *some* seed spawns rather than that a particular one does. A regression that breaks the
    /// commit path fails every seed and so fails this test; a regression that merely shifts the
    /// draw sequence does not.
    #[test]
    fn the_spawn_loop_puts_live_creatures_in_the_world() {
        init_vanilla_registry();
        init_behaviors();
        init_globals_once();
        let world = fresh_test_world("spawn_loop_end_to_end");

        // Far from the world's respawn point, so the 24-block exclusion cannot mask a failure.
        let chunk = ChunkPos::new(10, 10);
        insert_ready_full_chunk(&world, chunk);
        lay_grass_floor(&world, chunk, GRASS_FLOOR_Y);

        // Between the 24-block exclusion and the creature despawn distance, so the position tests
        // accept the column while the player still enables it.
        let player = joined_player(
            &world,
            "SpawnLoopWitness",
            chunk_center(10, 10) + DVec3::new(60.0, 0.0, 0.0),
        );

        let before = creature_count(&world);
        let mut spawned = 0_usize;
        for seed in 0..END_TO_END_SEEDS {
            let players = world.chunk_map.spawning_players();
            let mut state = create_state(
                &world,
                u32::try_from(players.natural_spawn_chunk_count()).unwrap_or(u32::MAX),
                &players,
            );
            let mut random = LegacyRandom::from_seed(seed);

            spawn_for_chunk(
                &world,
                chunk,
                &mut state,
                &[MobCategory::Creature],
                &mut random,
            );

            spawned = creature_count(&world) - before;
            if spawned > 0 {
                break;
            }
        }

        assert!(
            spawned > 0,
            "no creature spawned across {END_TO_END_SEEDS} seeds; the commit path is broken"
        );

        world.remove_player_for_world_change(&player);
    }
}
