//! Vanilla's `SpawnPlacements` registration table and the predicates it dispatches to.
//!
//! Vanilla keys a map from entity type onto a `(heightmap, placement, predicate)` triple, filled by
//! 83 registrations in a single static initialiser. An unregistered type is not an error: it falls
//! back to `NO_RESTRICTIONS`, the `MOTION_BLOCKING_NO_LEAVES` heightmap, and a predicate that always
//! passes. That fallback carries real weight, because most of the 158 entity types never register.
//!
//! The rows in [`PLACEMENTS`] are transcribed from that initialiser in its own order. Fifty of them
//! share seven predicates defined on vanilla's `Mob`, `Monster`, `Animal`, `WaterAnimal` and
//! `AgeableWaterCreature` base classes; those seven are ported here as free functions and selected
//! by [`SpawnPredicate`]. The remaining 33 rows name a predicate written on the mob's own class —
//! 32 distinct methods, since `Guardian`'s serves both guardians. Steel has a Rust entity class for
//! exactly one of them, `Endermite`, so that predicate is ported too and the other 32 rows carry
//! [`None`], which [`check_spawn_rules`] answers `false` for.
//!
//! `false` is the only safe reading of a missing predicate: it cannot invent a spawn vanilla would
//! refuse, and none of those 32 types can be constructed by Steel today in any case. The tests pin
//! that second half, so a Rust class landing without its predicate fails rather than spawns blind.

use std::ptr;

use glam::DVec3;
use steel_registry::blocks::block_state_ext::BlockStateExt as _;
use steel_registry::entity_type::EntityTypeRef;
use steel_registry::fluid::FluidState;
use steel_registry::vanilla_block_tags::BlockTag;
use steel_registry::vanilla_fluid_tags::FluidTag;
use steel_registry::{vanilla_blocks, vanilla_entities};
use steel_utils::BlockPos;
use steel_utils::BlockStateId;
use steel_utils::random::Random;
use steel_utils::types::Difficulty;

use crate::behavior::{BLOCK_BEHAVIORS, BlockStateBehaviorExt as _};
use crate::chunk::heightmap::HeightmapType;
use crate::chunk::light::LightLayer;
use crate::entity::EntitySpawnReason;
use crate::entity::ai::path::PathComputationType;
use crate::entity::block_danger::is_block_dangerous;
use crate::world::{LevelReader, ServerLevelAccessor, SignalQueryContext, is_redstone_conductor};

/// Denominator of vanilla `Ocelot.checkOcelotSpawnRules`' `random.nextInt(3) != 0`.
const OCELOT_SPAWN_DENOMINATOR: i32 = 3;

/// Vanilla `SpawnPlacementTypes`, the medium a mob's spawn position is tested against.
///
/// Vanilla writes these as four lambdas over `SpawnPlacementType`, three of which open with the same
/// world-border guard. Steel keeps them as one enum with the guard repeated per variant, matching
/// vanilla's own structure rather than hoisting it: [`Self::NoRestrictions`] deliberately does not
/// consult the border, and hoisting would either lose that or read as an oversight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpawnPlacementType {
    /// Accepts any position; vanilla's fallback for an unregistered type.
    NoRestrictions,
    /// Requires water at the position, as fish and squid do.
    InWater,
    /// Requires lava at the position; only the strider registers this.
    InLava,
    /// Requires a walkable surface below an unobstructed position.
    OnGround,
}

impl SpawnPlacementType {
    /// Returns vanilla `SpawnPlacementType.isSpawnPositionOk`.
    ///
    /// Vanilla marks the entity type `@Nullable` and answers `false` from the three restricted
    /// variants when it is absent, while [`Self::NoRestrictions`] answers `true` without looking. All
    /// nine of vanilla's call sites pass a type, so Steel takes one by value and that dead branch
    /// disappears.
    ///
    /// Takes [`ServerLevelAccessor`] rather than vanilla's `LevelReader` only because Steel keeps the
    /// world border on that trait; see [`ServerLevelAccessor::is_block_within_world_border`].
    #[must_use]
    pub fn is_spawn_position_ok(
        self,
        level: &dyn ServerLevelAccessor,
        pos: BlockPos,
        entity_type: EntityTypeRef,
    ) -> bool {
        match self {
            Self::NoRestrictions => true,
            Self::InWater => {
                if !level.is_block_within_world_border(pos) {
                    return false;
                }

                let above = pos.above();
                level
                    .get_block_state(pos)
                    .get_fluid_state()
                    .fluid_id
                    .has_tag(&FluidTag::WATER)
                    && !is_redstone_conductor(level, level.get_block_state(above), above)
            }
            Self::InLava => {
                level.is_block_within_world_border(pos)
                    && level
                        .get_block_state(pos)
                        .get_fluid_state()
                        .fluid_id
                        .has_tag(&FluidTag::LAVA)
            }
            Self::OnGround => {
                if !level.is_block_within_world_border(pos) {
                    return false;
                }

                let below = pos.below();
                level.is_valid_spawn(level.get_block_state(below), below, entity_type)
                    && is_valid_empty_spawn_block_at(level, pos, entity_type)
                    && is_valid_empty_spawn_block_at(level, pos.above(), entity_type)
            }
        }
    }

    /// Returns vanilla `SpawnPlacementType.adjustSpawnPosition`.
    ///
    /// [`Self::OnGround`] is the only variant that overrides vanilla's default, which returns the
    /// candidate untouched. It steps one block down when that block can be walked through, so a mob
    /// picked at a heightmap surface stands on the ground rather than inside a plant growing on it.
    #[must_use]
    pub fn adjust_spawn_position(self, level: &dyn LevelReader, candidate: BlockPos) -> BlockPos {
        if self != Self::OnGround {
            return candidate;
        }

        let below = candidate.below();
        if level
            .get_block_state(below)
            .is_pathfindable(PathComputationType::Land)
        {
            below
        } else {
            candidate
        }
    }
}

/// Returns vanilla `NaturalSpawner.isValidEmptySpawnBlock`.
///
/// Vanilla's home for this is the spawner rather than the placement table, and it will be
/// re-exported from there once Steel has that module; [`SpawnPlacementType::OnGround`] is its only
/// caller today.
///
/// The fluid state is a parameter rather than a read because vanilla's other caller already holds
/// one. Steel derives it from the block state, which carries waterlogging, so the two always agree.
#[must_use]
pub fn is_valid_empty_spawn_block(
    level: &dyn LevelReader,
    pos: BlockPos,
    state: BlockStateId,
    fluid_state: FluidState,
    entity_type: EntityTypeRef,
) -> bool {
    !level.is_collision_shape_full_block(state, pos)
        && !BLOCK_BEHAVIORS
            .get_behavior(state.get_block())
            .is_signal_source(state, SignalQueryContext::DEFAULT)
        && fluid_state.is_empty()
        && !state
            .get_block()
            .has_tag(&BlockTag::PREVENT_MOB_SPAWNING_INSIDE)
        && !is_block_dangerous(entity_type, state)
}

/// Reads the state at a position and tests it, as vanilla's private overload of the same name does.
fn is_valid_empty_spawn_block_at(
    level: &dyn LevelReader,
    pos: BlockPos,
    entity_type: EntityTypeRef,
) -> bool {
    let state = level.get_block_state(pos);
    is_valid_empty_spawn_block(level, pos, state, state.get_fluid_state(), entity_type)
}

/// Which shared vanilla spawn predicate a registered entity type carries.
///
/// Vanilla stores a method reference per row. Steel names the method instead, so the table stays a
/// plain `static` and the eight bodies stay ordinary functions that tests can call directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpawnPredicate {
    /// Vanilla `Mob.checkMobSpawnRules`.
    Mob,
    /// Vanilla `Monster.checkMonsterSpawnRules`.
    Monster,
    /// Vanilla `Monster.checkAnyLightMonsterSpawnRules`.
    AnyLightMonster,
    /// Vanilla `Monster.checkSurfaceMonstersSpawnRules`.
    SurfaceMonster,
    /// Vanilla `Animal.checkAnimalSpawnRules`.
    Animal,
    /// Vanilla `WaterAnimal.checkSurfaceWaterAnimalSpawnRules`.
    SurfaceWaterAnimal,
    /// Vanilla `AgeableWaterCreature.checkSurfaceAgeableWaterCreatureSpawnRules`.
    ///
    /// Vanilla's two water predicates are textually identical and differ only in the class they sit
    /// on and the type bound they accept. They are kept apart here so the table records which row
    /// took which, and share one match arm below because their bodies really are the same.
    SurfaceAgeableWaterCreature,
    /// Vanilla `Endermite.checkEndermiteSpawnRules`.
    Endermite,
    /// Vanilla `Ocelot.checkOcelotSpawnRules`.
    ///
    /// The whole rule is `random.nextInt(3) != 0`, with no light, block or biome term at all: an
    /// ocelot's biome table already confines it to jungles, and this only thins it to two attempts
    /// in three once it gets there. Note it draws from the random source unconditionally, so it
    /// consumes randomness even on the attempts it refuses.
    Ocelot,
    /// Vanilla `MushroomCow.checkMushroomSpawnRules`.
    MushroomCow,
}

impl SpawnPredicate {
    /// Evaluates this predicate, mirroring vanilla's `SpawnPredicate.test`.
    #[must_use]
    pub fn test(
        self,
        entity_type: EntityTypeRef,
        level: &dyn ServerLevelAccessor,
        spawn_reason: EntitySpawnReason,
        pos: BlockPos,
        random: &mut impl Random,
    ) -> bool {
        match self {
            Self::Mob => check_mob_spawn_rules(entity_type, level, spawn_reason, pos),
            Self::Monster => {
                check_monster_spawn_rules(entity_type, level, spawn_reason, pos, random)
            }
            Self::AnyLightMonster => {
                check_any_light_monster_spawn_rules(entity_type, level, spawn_reason, pos)
            }
            Self::SurfaceMonster => {
                check_surface_monsters_spawn_rules(entity_type, level, spawn_reason, pos, random)
            }
            Self::Animal => check_animal_spawn_rules(level, spawn_reason, pos),
            Self::SurfaceWaterAnimal | Self::SurfaceAgeableWaterCreature => {
                check_surface_water_animal_spawn_rules(level, pos)
            }
            Self::Endermite => check_endermite_spawn_rules(entity_type, level, spawn_reason, pos),
            Self::Ocelot => check_ocelot_spawn_rules(random),
            Self::MushroomCow => check_mushroom_spawn_rules(level, pos),
        }
    }
}

/// Returns vanilla `Mob.checkMobSpawnRules`.
///
/// The block one below the candidate position decides, through its own `isValidSpawn` predicate.
/// A spawner bypasses the question entirely.
#[must_use]
pub fn check_mob_spawn_rules(
    entity_type: EntityTypeRef,
    level: &dyn LevelReader,
    spawn_reason: EntitySpawnReason,
    pos: BlockPos,
) -> bool {
    let below = pos.below();
    spawn_reason.is_spawner()
        || level.is_valid_spawn(level.get_block_state(below), below, entity_type)
}

/// Returns vanilla `Monster.isDarkEnoughToSpawn`.
///
/// Three gates in vanilla's order, and the order is what makes the random draws line up: the sky
/// light must lose a roll against `nextInt(32)`, the block light must sit within the dimension's
/// limit, and the combined brightness must not exceed a sample of the dimension's spawn light test.
/// Only the first and last consume randomness, and the last is skipped when either earlier gate
/// fails, exactly as the short-circuit in the source does.
#[must_use]
pub fn is_dark_enough_to_spawn(
    level: &dyn ServerLevelAccessor,
    pos: BlockPos,
    random: &mut impl Random,
) -> bool {
    if i32::from(level.brightness(LightLayer::Sky, pos)) > random.next_i32_bounded(32) {
        return false;
    }

    let dimension_type = level.dimension_type();
    let block_light_limit = dimension_type.monster_spawn_block_light_limit;
    if block_light_limit < 15
        && i32::from(level.brightness(LightLayer::Block, pos)) > block_light_limit
    {
        return false;
    }

    let brightness = if level.is_thundering() {
        level.max_local_raw_brightness(pos, 10)
    } else {
        level.max_local_raw_brightness_now(pos)
    };
    i32::from(brightness) <= dimension_type.monster_spawn_light_level.sample(random)
}

/// Returns vanilla `Monster.checkMonsterSpawnRules`.
#[must_use]
pub fn check_monster_spawn_rules(
    entity_type: EntityTypeRef,
    level: &dyn ServerLevelAccessor,
    spawn_reason: EntitySpawnReason,
    pos: BlockPos,
    random: &mut impl Random,
) -> bool {
    (spawn_reason.ignores_light_requirements() || is_dark_enough_to_spawn(level, pos, random))
        && check_mob_spawn_rules(entity_type, level, spawn_reason, pos)
}

/// Returns vanilla `Monster.checkAnyLightMonsterSpawnRules`.
///
/// Vanilla's body is a bare call to `checkMobSpawnRules`; the separate name exists so the table can
/// record which monsters skip the darkness test. Blaze, breeze and zoglin are the three that do.
#[must_use]
pub fn check_any_light_monster_spawn_rules(
    entity_type: EntityTypeRef,
    level: &dyn LevelReader,
    spawn_reason: EntitySpawnReason,
    pos: BlockPos,
) -> bool {
    check_mob_spawn_rules(entity_type, level, spawn_reason, pos)
}

/// Returns vanilla `Monster.checkSurfaceMonstersSpawnRules`.
#[must_use]
pub fn check_surface_monsters_spawn_rules(
    entity_type: EntityTypeRef,
    level: &dyn ServerLevelAccessor,
    spawn_reason: EntitySpawnReason,
    pos: BlockPos,
    random: &mut impl Random,
) -> bool {
    check_monster_spawn_rules(entity_type, level, spawn_reason, pos, random)
        && (spawn_reason.is_spawner() || level.can_see_sky(pos))
}

/// Returns vanilla `Animal.isBrightEnoughToSpawn`.
#[must_use]
pub fn is_bright_enough_to_spawn(level: &dyn LevelReader, pos: BlockPos) -> bool {
    level.raw_brightness(pos, 0) > 8
}

/// Returns vanilla `Animal.checkAnimalSpawnRules`.
///
/// Unlike the monster predicates this one reads a block tag rather than the block's own spawn rule,
/// and in 26.2 `#minecraft:animals_spawnable_on` holds exactly one block, `grass_block`. So an
/// equally sturdy stone or dirt block admits no animals at all, which is what keeps herds on grass.
#[must_use]
pub fn check_animal_spawn_rules(
    level: &dyn LevelReader,
    spawn_reason: EntitySpawnReason,
    pos: BlockPos,
) -> bool {
    let bright_enough =
        spawn_reason.ignores_light_requirements() || is_bright_enough_to_spawn(level, pos);
    level
        .get_block_state(pos.below())
        .get_block()
        .has_tag(&BlockTag::ANIMALS_SPAWNABLE_ON)
        && bright_enough
}

/// Returns vanilla `MushroomCow.checkMushroomSpawnRules`.
///
/// The same shape as [`check_animal_spawn_rules`] against a different tag: mooshrooms want mycelium
/// rather than the grass and dirt every other animal accepts, which is what confines them to
/// mushroom fields without any biome term in the rule itself.
///
/// Vanilla writes `isBrightEnoughToSpawn` directly rather than going through the spawn-reason
/// escape the animal rule uses, so a mooshroom from a spawn egg still wants the light.
#[must_use]
pub fn check_mushroom_spawn_rules(level: &dyn LevelReader, pos: BlockPos) -> bool {
    level
        .get_block_state(pos.below())
        .get_block()
        .has_tag(&BlockTag::MOOSHROOMS_SPAWNABLE_ON)
        && is_bright_enough_to_spawn(level, pos)
}

/// Returns vanilla `WaterAnimal.checkSurfaceWaterAnimalSpawnRules`.
///
/// Also serves `AgeableWaterCreature.checkSurfaceAgeableWaterCreatureSpawnRules`, whose body is the
/// same. The window is the thirteen blocks below sea level plus sea level itself, and it wants water
/// both below and above, which is what keeps fish out of a one-block puddle.
#[must_use]
pub fn check_surface_water_animal_spawn_rules(
    level: &dyn ServerLevelAccessor,
    pos: BlockPos,
) -> bool {
    let sea_level = level.sea_level();
    let min_spawn_level = sea_level - 13;
    pos.y() >= min_spawn_level
        && pos.y() <= sea_level
        && level
            .get_block_state(pos.below())
            .get_fluid_state()
            .fluid_id
            .has_tag(&FluidTag::WATER)
        && level.get_block_state(pos.above()).get_block() == &vanilla_blocks::WATER
}

/// Returns vanilla `Endermite.checkEndermiteSpawnRules`.
///
/// Endermites refuse to spawn within five blocks of a player, which is the whole reason the
/// predicate exists: vanilla spawns them from a thrown ender pearl and this keeps them off the
/// thrower. The player query ignores creative-mode and spectating players.
#[must_use]
pub fn check_endermite_spawn_rules(
    entity_type: EntityTypeRef,
    level: &dyn ServerLevelAccessor,
    spawn_reason: EntitySpawnReason,
    pos: BlockPos,
) -> bool {
    if !check_any_light_monster_spawn_rules(entity_type, level, spawn_reason, pos) {
        return false;
    }

    if spawn_reason.is_spawner() {
        return true;
    }

    !level.has_nearby_non_creative_player(DVec3::from(pos.get_center()), 5.0)
}

/// Returns vanilla `Ocelot.checkOcelotSpawnRules`.
///
/// Two attempts in three succeed. Vanilla takes no position, level or reason argument at all — the
/// jungle restriction lives in the biome's spawner table, not here.
#[must_use]
pub fn check_ocelot_spawn_rules(random: &mut impl Random) -> bool {
    random.next_i32_bounded(OCELOT_SPAWN_DENOMINATOR) != 0
}

/// One row of vanilla's `SpawnPlacements` map.
#[derive(Debug)]
pub struct SpawnPlacementData {
    /// The entity type this row was registered for.
    pub entity_type: EntityTypeRef,
    /// The medium the spawn position is tested against.
    pub placement: SpawnPlacementType,
    /// The heightmap a spawn attempt picks its Y from.
    pub heightmap: HeightmapType,
    /// The registered predicate, or [`None`] where vanilla names one Steel has not ported.
    pub predicate: Option<SpawnPredicate>,
}

/// Returns the row registered for an entity type, if it has one.
///
/// Vanilla hashes; this scans and compares addresses, as [`steel_registry::blocks::spawn_rule`]
/// already does for the per-block rule. Eighty-three identity comparisons run once per spawn
/// candidate, which is cheaper than hashing an `Identifier`.
#[must_use]
pub fn placement_data(entity_type: EntityTypeRef) -> Option<&'static SpawnPlacementData> {
    PLACEMENTS
        .iter()
        .find(|data| ptr::eq(data.entity_type, entity_type))
}

/// Returns vanilla `SpawnPlacements.getPlacementType`.
///
/// An unregistered type answers [`SpawnPlacementType::NoRestrictions`].
#[must_use]
pub fn placement_type(entity_type: EntityTypeRef) -> SpawnPlacementType {
    placement_data(entity_type).map_or(SpawnPlacementType::NoRestrictions, |data| data.placement)
}

/// Returns vanilla `SpawnPlacements.isSpawnPositionOk`.
///
/// An unregistered type answers through [`SpawnPlacementType::NoRestrictions`], which accepts every
/// position — the world border included, since vanilla's lambda for that variant reads nothing.
#[must_use]
pub fn is_spawn_position_ok(
    entity_type: EntityTypeRef,
    level: &dyn ServerLevelAccessor,
    pos: BlockPos,
) -> bool {
    placement_type(entity_type).is_spawn_position_ok(level, pos, entity_type)
}

/// Returns vanilla `SpawnPlacements.getHeightmapType`.
///
/// An unregistered type answers [`HeightmapType::MotionBlockingNoLeaves`].
#[must_use]
pub fn heightmap_type(entity_type: EntityTypeRef) -> HeightmapType {
    placement_data(entity_type).map_or(HeightmapType::MotionBlockingNoLeaves, |data| data.heightmap)
}

/// Returns vanilla `SpawnPlacements.checkSpawnRules`.
///
/// The peaceful gate runs first and applies to every type, registered or not. After it, an
/// unregistered type passes; a registered one defers to its predicate. A row whose predicate Steel
/// has not ported answers `false`, which is where this deliberately stops short of vanilla rather
/// than guessing at it.
#[must_use]
pub fn check_spawn_rules(
    entity_type: EntityTypeRef,
    level: &dyn ServerLevelAccessor,
    spawn_reason: EntitySpawnReason,
    pos: BlockPos,
    random: &mut impl Random,
) -> bool {
    if !entity_type.allowed_in_peaceful && level.difficulty() == Difficulty::Peaceful {
        return false;
    }

    match placement_data(entity_type) {
        None => true,
        Some(data) => data
            .predicate
            .is_some_and(|predicate| predicate.test(entity_type, level, spawn_reason, pos, random)),
    }
}

/// Expands one [`PLACEMENTS`] predicate cell into [`SpawnPlacementData::predicate`].
///
/// `None` spells a row whose vanilla predicate is written on the mob's own class and has not been
/// ported; any other token names a [`SpawnPredicate`] variant.
macro_rules! spawn_predicate {
    (None) => {
        None
    };
    ($variant:ident) => {
        Some(SpawnPredicate::$variant)
    };
}

/// Builds [`PLACEMENTS`] from one row per vanilla `SpawnPlacements.register` call.
///
/// A row carries that call's four arguments in vanilla's own order, so the table can be diffed
/// against the Java initialiser line by line.
macro_rules! placements {
    ($($entity:ident => $placement:ident, $heightmap:ident, $predicate:ident;)*) => {
        &[$(SpawnPlacementData {
            entity_type: &vanilla_entities::$entity,
            placement: SpawnPlacementType::$placement,
            heightmap: HeightmapType::$heightmap,
            predicate: spawn_predicate!($predicate),
        }),*]
    };
}

/// Vanilla's `SpawnPlacements` static initialiser, all 83 registrations in registration order.
static PLACEMENTS: &[SpawnPlacementData] = placements! {
    AXOLOTL => InWater, MotionBlockingNoLeaves, None;
    COD => InWater, MotionBlockingNoLeaves, SurfaceWaterAnimal;
    DOLPHIN => InWater, MotionBlockingNoLeaves, SurfaceAgeableWaterCreature;
    DROWNED => InWater, MotionBlockingNoLeaves, None;
    GUARDIAN => InWater, MotionBlockingNoLeaves, None;
    PUFFERFISH => InWater, MotionBlockingNoLeaves, SurfaceWaterAnimal;
    SALMON => InWater, MotionBlockingNoLeaves, SurfaceWaterAnimal;
    SQUID => InWater, MotionBlockingNoLeaves, SurfaceAgeableWaterCreature;
    TROPICAL_FISH => InWater, MotionBlockingNoLeaves, None;
    ARMADILLO => OnGround, MotionBlockingNoLeaves, None;
    BAT => OnGround, MotionBlockingNoLeaves, None;
    BLAZE => OnGround, MotionBlockingNoLeaves, AnyLightMonster;
    BOGGED => OnGround, MotionBlockingNoLeaves, Monster;
    BREEZE => OnGround, MotionBlockingNoLeaves, AnyLightMonster;
    CAMEL => OnGround, MotionBlockingNoLeaves, None;
    CAMEL_HUSK => OnGround, MotionBlockingNoLeaves, SurfaceMonster;
    CAVE_SPIDER => OnGround, MotionBlockingNoLeaves, Monster;
    CHICKEN => OnGround, MotionBlockingNoLeaves, Animal;
    COW => OnGround, MotionBlockingNoLeaves, Animal;
    CREEPER => OnGround, MotionBlockingNoLeaves, Monster;
    DONKEY => OnGround, MotionBlockingNoLeaves, Animal;
    ENDERMAN => OnGround, MotionBlockingNoLeaves, Monster;
    ENDERMITE => OnGround, MotionBlockingNoLeaves, Endermite;
    ENDER_DRAGON => OnGround, MotionBlockingNoLeaves, Mob;
    FROG => OnGround, MotionBlockingNoLeaves, None;
    GHAST => OnGround, MotionBlockingNoLeaves, None;
    HAPPY_GHAST => OnGround, MotionBlockingNoLeaves, Animal;
    GIANT => OnGround, MotionBlockingNoLeaves, Monster;
    GLOW_SQUID => InWater, MotionBlockingNoLeaves, None;
    GOAT => OnGround, MotionBlockingNoLeaves, None;
    HORSE => OnGround, MotionBlockingNoLeaves, Animal;
    HUSK => OnGround, MotionBlockingNoLeaves, SurfaceMonster;
    IRON_GOLEM => OnGround, MotionBlockingNoLeaves, Mob;
    LLAMA => OnGround, MotionBlockingNoLeaves, Animal;
    MAGMA_CUBE => OnGround, MotionBlockingNoLeaves, None;
    SULFUR_CUBE => OnGround, MotionBlockingNoLeaves, None;
    MOOSHROOM => OnGround, MotionBlockingNoLeaves, MushroomCow;
    MULE => OnGround, MotionBlockingNoLeaves, Animal;
    NAUTILUS => InWater, MotionBlockingNoLeaves, None;
    OCELOT => OnGround, MotionBlocking, Ocelot;
    PARROT => OnGround, MotionBlocking, None;
    PIG => OnGround, MotionBlockingNoLeaves, Animal;
    HOGLIN => OnGround, MotionBlockingNoLeaves, None;
    PIGLIN => OnGround, MotionBlockingNoLeaves, None;
    PILLAGER => OnGround, MotionBlockingNoLeaves, None;
    POLAR_BEAR => OnGround, MotionBlockingNoLeaves, None;
    RABBIT => OnGround, MotionBlockingNoLeaves, None;
    SHEEP => OnGround, MotionBlockingNoLeaves, Animal;
    SILVERFISH => OnGround, MotionBlockingNoLeaves, None;
    SKELETON => OnGround, MotionBlockingNoLeaves, Monster;
    SKELETON_HORSE => OnGround, MotionBlockingNoLeaves, None;
    SLIME => OnGround, MotionBlockingNoLeaves, None;
    SNOW_GOLEM => OnGround, MotionBlockingNoLeaves, Mob;
    SPIDER => OnGround, MotionBlockingNoLeaves, Monster;
    STRAY => OnGround, MotionBlockingNoLeaves, None;
    PARCHED => OnGround, MotionBlockingNoLeaves, SurfaceMonster;
    STRIDER => InLava, MotionBlockingNoLeaves, None;
    TURTLE => OnGround, MotionBlockingNoLeaves, None;
    VILLAGER => OnGround, MotionBlockingNoLeaves, Mob;
    WITCH => OnGround, MotionBlockingNoLeaves, Monster;
    WITHER => OnGround, MotionBlockingNoLeaves, Monster;
    WITHER_SKELETON => OnGround, MotionBlockingNoLeaves, Monster;
    WOLF => OnGround, MotionBlockingNoLeaves, None;
    ZOGLIN => OnGround, MotionBlockingNoLeaves, AnyLightMonster;
    CREAKING => OnGround, MotionBlockingNoLeaves, Monster;
    ZOMBIE => OnGround, MotionBlockingNoLeaves, Monster;
    ZOMBIE_HORSE => OnGround, MotionBlockingNoLeaves, Monster;
    ZOMBIFIED_PIGLIN => OnGround, MotionBlockingNoLeaves, None;
    ZOMBIE_VILLAGER => OnGround, MotionBlockingNoLeaves, Monster;
    CAT => OnGround, MotionBlockingNoLeaves, Animal;
    ELDER_GUARDIAN => InWater, MotionBlockingNoLeaves, None;
    EVOKER => NoRestrictions, MotionBlockingNoLeaves, Monster;
    FOX => NoRestrictions, MotionBlockingNoLeaves, None;
    ILLUSIONER => NoRestrictions, MotionBlockingNoLeaves, Monster;
    PANDA => NoRestrictions, MotionBlockingNoLeaves, Animal;
    PHANTOM => NoRestrictions, MotionBlockingNoLeaves, Mob;
    RAVAGER => OnGround, MotionBlockingNoLeaves, Monster;
    SHULKER => NoRestrictions, MotionBlockingNoLeaves, Mob;
    TRADER_LLAMA => NoRestrictions, MotionBlockingNoLeaves, Animal;
    VEX => NoRestrictions, MotionBlockingNoLeaves, Monster;
    VINDICATOR => NoRestrictions, MotionBlockingNoLeaves, Monster;
    WANDERING_TRADER => OnGround, MotionBlockingNoLeaves, Mob;
    WARDEN => NoRestrictions, MotionBlockingNoLeaves, Monster;
};

#[cfg(test)]
mod tests {
    use steel_registry::init_vanilla_registry;
    use steel_utils::random::xoroshiro::Xoroshiro;

    use super::*;
    use crate::behavior::init_behaviors;
    use crate::entity::{ENTITIES, init_entities};
    use crate::test_support::TestLevel;

    /// The position a spawn is tested at; the block that decides sits one below it.
    const SPAWN_POS: BlockPos = BlockPos::new(0, 64, 0);

    fn count_placement(placement: SpawnPlacementType) -> usize {
        PLACEMENTS
            .iter()
            .filter(|data| data.placement == placement)
            .count()
    }

    fn count_heightmap(heightmap: HeightmapType) -> usize {
        PLACEMENTS
            .iter()
            .filter(|data| data.heightmap == heightmap)
            .count()
    }

    fn count_predicate(predicate: Option<SpawnPredicate>) -> usize {
        PLACEMENTS
            .iter()
            .filter(|data| data.predicate == predicate)
            .count()
    }

    /// Vanilla's `SpawnPlacements` initialiser holds exactly 83 `register` calls.
    #[test]
    fn table_holds_every_vanilla_registration() {
        assert_eq!(PLACEMENTS.len(), 83);
    }

    /// Pins the medium each of the 83 rows was registered against.
    ///
    /// Only the strider is registered `IN_LAVA`, and getting one of these wrong silently relocates a
    /// mob rather than failing anything, which is why the whole distribution is asserted.
    #[test]
    fn placement_distribution_matches_vanilla() {
        assert_eq!(count_placement(SpawnPlacementType::OnGround), 60);
        assert_eq!(count_placement(SpawnPlacementType::InWater), 12);
        assert_eq!(count_placement(SpawnPlacementType::NoRestrictions), 10);
        assert_eq!(count_placement(SpawnPlacementType::InLava), 1);
    }

    /// Only the ocelot and the parrot pick their Y off the leaf-carrying heightmap.
    #[test]
    fn heightmap_distribution_matches_vanilla() {
        assert_eq!(count_heightmap(HeightmapType::MotionBlockingNoLeaves), 81);
        assert_eq!(count_heightmap(HeightmapType::MotionBlocking), 2);

        for entity_type in [&vanilla_entities::OCELOT, &vanilla_entities::PARROT] {
            assert_eq!(
                heightmap_type(entity_type),
                HeightmapType::MotionBlocking,
                "{} is one of vanilla's two MOTION_BLOCKING rows",
                entity_type.key
            );
        }
    }

    /// Pins which predicate each row took, including the 32 vanilla names Steel has not ported.
    ///
    /// The `None` count is the interesting one: it is the exact size of the gap this module
    /// deliberately leaves, so porting a predicate without deleting its `None` fails here.
    #[test]
    fn predicate_distribution_matches_vanilla() {
        assert_eq!(count_predicate(Some(SpawnPredicate::Monster)), 20);
        assert_eq!(count_predicate(Some(SpawnPredicate::Animal)), 12);
        assert_eq!(count_predicate(Some(SpawnPredicate::Mob)), 7);
        assert_eq!(count_predicate(Some(SpawnPredicate::AnyLightMonster)), 3);
        assert_eq!(count_predicate(Some(SpawnPredicate::SurfaceMonster)), 3);
        assert_eq!(count_predicate(Some(SpawnPredicate::SurfaceWaterAnimal)), 3);
        assert_eq!(
            count_predicate(Some(SpawnPredicate::SurfaceAgeableWaterCreature)),
            2
        );
        assert_eq!(count_predicate(Some(SpawnPredicate::Endermite)), 1);
        assert_eq!(count_predicate(Some(SpawnPredicate::Ocelot)), 1);
        assert_eq!(count_predicate(Some(SpawnPredicate::MushroomCow)), 1);
        assert_eq!(count_predicate(None), 30);
    }

    /// Vanilla's map would silently keep the last write; a duplicated row here would be dead weight.
    #[test]
    fn no_entity_type_is_registered_twice() {
        for (index, data) in PLACEMENTS.iter().enumerate() {
            assert!(
                !PLACEMENTS
                    .iter()
                    .skip(index + 1)
                    .any(|other| ptr::eq(other.entity_type, data.entity_type)),
                "{} is registered more than once",
                data.entity_type.key
            );
        }
    }

    /// An unregistered type takes vanilla's fallback row and passes the rules outright.
    ///
    /// Most of the 158 entity types never register, so this branch carries more traffic than the
    /// table does. The armor stand is one of them.
    #[test]
    fn unregistered_types_take_vanillas_fallback_row() {
        init_vanilla_registry();
        let level = TestLevel::default();
        let mut random = Xoroshiro::from_seed(1);

        assert!(placement_data(&vanilla_entities::ARMOR_STAND).is_none());
        assert_eq!(
            placement_type(&vanilla_entities::ARMOR_STAND),
            SpawnPlacementType::NoRestrictions
        );
        assert_eq!(
            heightmap_type(&vanilla_entities::ARMOR_STAND),
            HeightmapType::MotionBlockingNoLeaves
        );
        assert!(check_spawn_rules(
            &vanilla_entities::ARMOR_STAND,
            &level,
            EntitySpawnReason::Natural,
            SPAWN_POS,
            &mut random,
        ));
    }

    /// A registered type answers its own row rather than the fallback.
    #[test]
    fn registered_rows_answer_their_own_placement_and_heightmap() {
        assert_eq!(
            placement_type(&vanilla_entities::STRIDER),
            SpawnPlacementType::InLava
        );
        assert_eq!(
            placement_type(&vanilla_entities::COD),
            SpawnPlacementType::InWater
        );
        assert_eq!(
            placement_type(&vanilla_entities::WARDEN),
            SpawnPlacementType::NoRestrictions
        );
        assert_eq!(
            placement_type(&vanilla_entities::ZOMBIE),
            SpawnPlacementType::OnGround
        );
    }

    /// No row that answers `false` for want of a predicate is a type Steel can actually construct.
    ///
    /// This is what makes the 32 [`None`] cells safe rather than merely convenient, and it is the
    /// assertion that breaks when a Rust entity class lands without its vanilla spawn predicate.
    /// The endermite is asserted from the other side, so the check cannot pass by being vacuous.
    #[test]
    fn unported_predicate_rows_have_no_rust_entity_class() {
        init_vanilla_registry();
        init_entities();

        for data in PLACEMENTS.iter().filter(|data| data.predicate.is_none()) {
            assert!(
                !ENTITIES.has_factory(data.entity_type),
                "{} has a Rust entity class but no ported spawn predicate",
                data.entity_type.key
            );
        }

        assert!(
            ENTITIES.has_factory(&vanilla_entities::ENDERMITE),
            "the endermite must have a factory, or this test proves nothing"
        );
    }

    /// A spawner skips the block-below question entirely, which is why a spawner block works in
    /// midair.
    #[test]
    fn spawners_bypass_the_block_below_check() {
        init_vanilla_registry();
        let level = TestLevel::default();

        assert!(!check_mob_spawn_rules(
            &vanilla_entities::ZOMBIE,
            &level,
            EntitySpawnReason::Natural,
            SPAWN_POS
        ));
        assert!(check_mob_spawn_rules(
            &vanilla_entities::ZOMBIE,
            &level,
            EntitySpawnReason::Spawner,
            SPAWN_POS
        ));
    }

    /// Animals want the tagged block below and raw brightness strictly above 8.
    #[test]
    fn animal_rules_read_the_block_tag_and_the_light() {
        init_vanilla_registry();
        let grass = vanilla_blocks::GRASS_BLOCK.default_state();
        let below = SPAWN_POS.below();

        let on_grass = TestLevel::default()
            .with_block(below, grass)
            .with_raw_brightness(9);
        assert!(check_animal_spawn_rules(
            &on_grass,
            EntitySpawnReason::Natural,
            SPAWN_POS
        ));

        let on_stone = TestLevel::default()
            .with_block(below, vanilla_blocks::STONE.default_state())
            .with_raw_brightness(9);
        assert!(!check_animal_spawn_rules(
            &on_stone,
            EntitySpawnReason::Natural,
            SPAWN_POS
        ));

        let in_shade = TestLevel::default()
            .with_block(below, grass)
            .with_raw_brightness(8);
        assert!(!check_animal_spawn_rules(
            &in_shade,
            EntitySpawnReason::Natural,
            SPAWN_POS
        ));
        assert!(check_animal_spawn_rules(
            &in_shade,
            EntitySpawnReason::TrialSpawner,
            SPAWN_POS
        ));
    }

    /// The overworld's `monster_spawn_block_light_limit` is 0, which makes this gate deterministic.
    ///
    /// A single point of block light therefore refuses every monster no matter how the rolls fall,
    /// which is the whole reason a torch works.
    #[test]
    fn darkness_gate_honors_the_dimensions_block_light_limit() {
        init_vanilla_registry();
        let unlit = TestLevel::default().with_layer_brightness(LightLayer::Block, 0);
        let torchlit = TestLevel::default().with_layer_brightness(LightLayer::Block, 1);
        let mut random = Xoroshiro::from_seed(0xDA12_1234);

        for _ in 0..256 {
            assert!(is_dark_enough_to_spawn(&unlit, SPAWN_POS, &mut random));
            assert!(!is_dark_enough_to_spawn(&torchlit, SPAWN_POS, &mut random));
        }
    }

    /// The sky-light gate is a roll against `nextInt(32)` rather than a fixed threshold.
    ///
    /// So a spot lit to sky level 15 is not forbidden, only unlikely, and asserting a mixed outcome
    /// is what distinguishes the roll from a comparison against 15.
    #[test]
    fn sky_light_gate_is_a_roll_rather_than_a_threshold() {
        init_vanilla_registry();
        let level = TestLevel::default().with_layer_brightness(LightLayer::Sky, 15);
        let mut random = Xoroshiro::from_seed(0x5EED_5EED);

        let mut dark_enough = 0_u32;
        for _ in 0..256 {
            if is_dark_enough_to_spawn(&level, SPAWN_POS, &mut random) {
                dark_enough += 1;
            }
        }

        assert!(
            dark_enough > 0 && dark_enough < 256,
            "sky light 15 must sometimes win and sometimes lose the roll, got {dark_enough}/256"
        );
    }

    /// The surface water window is sea level and the thirteen blocks below it, water above and below.
    #[test]
    fn surface_water_window_matches_vanilla() {
        init_vanilla_registry();
        let level = TestLevel::default();
        for y in 40..=70 {
            level.set_test_block(
                BlockPos::new(0, y, 0),
                vanilla_blocks::WATER.default_state(),
            );
        }

        for y in [63, 50] {
            assert!(
                check_surface_water_animal_spawn_rules(&level, BlockPos::new(0, y, 0)),
                "y={y} is inside vanilla's window"
            );
        }
        for y in [64, 49] {
            assert!(
                !check_surface_water_animal_spawn_rules(&level, BlockPos::new(0, y, 0)),
                "y={y} is outside vanilla's window"
            );
        }

        level.set_test_block(BlockPos::new(0, 64, 0), vanilla_blocks::AIR.default_state());
        assert!(
            !check_surface_water_animal_spawn_rules(&level, BlockPos::new(0, 63, 0)),
            "water below is not enough; vanilla wants water above too"
        );
    }

    /// The peaceful gate runs before the predicate and reads the type's own `allowed_in_peaceful`.
    ///
    /// The zombie is refused on Peaceful in a spot it would otherwise accept, and the cow is not,
    /// which pins both the gate and its ordering ahead of the predicate.
    #[test]
    fn peaceful_gate_runs_before_the_predicate() {
        init_vanilla_registry();
        let below = SPAWN_POS.below();
        let stone = vanilla_blocks::STONE.default_state();
        let mut random = Xoroshiro::from_seed(7);

        let normal = TestLevel::default().with_block(below, stone);
        assert!(check_spawn_rules(
            &vanilla_entities::ZOMBIE,
            &normal,
            EntitySpawnReason::Natural,
            SPAWN_POS,
            &mut random
        ));

        let peaceful = TestLevel::default()
            .with_block(below, stone)
            .with_difficulty(Difficulty::Peaceful);
        assert!(!check_spawn_rules(
            &vanilla_entities::ZOMBIE,
            &peaceful,
            EntitySpawnReason::Natural,
            SPAWN_POS,
            &mut random
        ));

        let pasture = TestLevel::default()
            .with_block(below, vanilla_blocks::GRASS_BLOCK.default_state())
            .with_raw_brightness(9)
            .with_difficulty(Difficulty::Peaceful);
        assert!(check_spawn_rules(
            &vanilla_entities::COW,
            &pasture,
            EntitySpawnReason::Natural,
            SPAWN_POS,
            &mut random
        ));
    }

    /// Names which of [`is_valid_empty_spawn_block`]'s five terms a state fails, in vanilla's order.
    ///
    /// Each stand-in block below is chosen to fail exactly one, so the terms can be pinned
    /// individually against a predicate that only ever answers one bool.
    fn failing_spawn_block_terms(level: &TestLevel, state: BlockStateId) -> Vec<&'static str> {
        let mut failed = Vec::new();

        if level.is_collision_shape_full_block(state, SPAWN_POS) {
            failed.push("full collision shape");
        }
        if BLOCK_BEHAVIORS
            .get_behavior(state.get_block())
            .is_signal_source(state, SignalQueryContext::DEFAULT)
        {
            failed.push("signal source");
        }
        if !state.get_fluid_state().is_empty() {
            failed.push("fluid");
        }
        if state
            .get_block()
            .has_tag(&BlockTag::PREVENT_MOB_SPAWNING_INSIDE)
        {
            failed.push("prevent_mob_spawning_inside");
        }
        if is_block_dangerous(&vanilla_entities::ZOMBIE, state) {
            failed.push("hazard");
        }

        failed
    }

    /// Vanilla's `NO_RESTRICTIONS` lambda is `(level, pos, type) -> true` and reads nothing at all.
    ///
    /// Not even the world border, which is why the guard is repeated in the other three arms rather
    /// than hoisted above the match.
    #[test]
    fn no_restrictions_accepts_every_position() {
        init_vanilla_registry();
        let hostile = TestLevel::default()
            .with_default_block_state(vanilla_blocks::LAVA.default_state())
            .with_blocks_outside_world_border();

        assert!(SpawnPlacementType::NoRestrictions.is_spawn_position_ok(
            &hostile,
            SPAWN_POS,
            &vanilla_entities::PHANTOM
        ));
    }

    /// `IN_WATER` wants water at the position under a block that does not conduct redstone.
    #[test]
    fn in_water_wants_water_under_a_non_conducting_block() {
        init_vanilla_registry();
        init_behaviors();
        let water = vanilla_blocks::WATER.default_state();
        let cod = &vanilla_entities::COD;

        let open_water = TestLevel::default().with_block(SPAWN_POS, water);
        assert!(SpawnPlacementType::InWater.is_spawn_position_ok(&open_water, SPAWN_POS, cod));

        let capped = TestLevel::default()
            .with_block(SPAWN_POS, water)
            .with_block(SPAWN_POS.above(), vanilla_blocks::STONE.default_state());
        assert!(
            !SpawnPlacementType::InWater.is_spawn_position_ok(&capped, SPAWN_POS, cod),
            "a redstone conductor above refuses the spawn"
        );

        let dry = TestLevel::default();
        assert!(!SpawnPlacementType::InWater.is_spawn_position_ok(&dry, SPAWN_POS, cod));

        let lava = TestLevel::default().with_block(SPAWN_POS, vanilla_blocks::LAVA.default_state());
        assert!(
            !SpawnPlacementType::InWater.is_spawn_position_ok(&lava, SPAWN_POS, cod),
            "the guard is #water, not merely any fluid"
        );
    }

    /// `IN_LAVA` wants lava and, unlike `IN_WATER`, never looks up.
    #[test]
    fn in_lava_wants_lava_and_ignores_the_block_above() {
        init_vanilla_registry();
        let strider = &vanilla_entities::STRIDER;

        let capped = TestLevel::default()
            .with_block(SPAWN_POS, vanilla_blocks::LAVA.default_state())
            .with_block(SPAWN_POS.above(), vanilla_blocks::STONE.default_state());
        assert!(
            SpawnPlacementType::InLava.is_spawn_position_ok(&capped, SPAWN_POS, strider),
            "vanilla's IN_LAVA lambda has no conductor term, so lava under stone still spawns"
        );

        let water =
            TestLevel::default().with_block(SPAWN_POS, vanilla_blocks::WATER.default_state());
        assert!(!SpawnPlacementType::InLava.is_spawn_position_ok(&water, SPAWN_POS, strider));

        let dry = TestLevel::default();
        assert!(!SpawnPlacementType::InLava.is_spawn_position_ok(&dry, SPAWN_POS, strider));
    }

    /// `ON_GROUND` wants a valid spawn block below two positions that are empty spawn blocks.
    #[test]
    fn on_ground_wants_a_floor_under_two_clear_blocks() {
        init_vanilla_registry();
        init_behaviors();
        let stone = vanilla_blocks::STONE.default_state();
        let zombie = &vanilla_entities::ZOMBIE;

        let floor = TestLevel::default().with_block(SPAWN_POS.below(), stone);
        assert!(SpawnPlacementType::OnGround.is_spawn_position_ok(&floor, SPAWN_POS, zombie));

        let midair = TestLevel::default();
        assert!(
            !SpawnPlacementType::OnGround.is_spawn_position_ok(&midair, SPAWN_POS, zombie),
            "air below is not a valid spawn block"
        );

        for blocked in [SPAWN_POS, SPAWN_POS.above()] {
            let obstructed = TestLevel::default()
                .with_block(SPAWN_POS.below(), stone)
                .with_block(blocked, stone);
            assert!(
                !SpawnPlacementType::OnGround.is_spawn_position_ok(&obstructed, SPAWN_POS, zombie),
                "vanilla wants both {SPAWN_POS:?} and the block above it clear"
            );
        }
    }

    /// The world border is the first term of all three restricted variants.
    ///
    /// Each row is a position the variant otherwise accepts, so the refusal can only be the border.
    #[test]
    fn the_world_border_refuses_the_three_restricted_variants() {
        init_vanilla_registry();
        init_behaviors();
        let zombie = &vanilla_entities::ZOMBIE;
        let air = vanilla_blocks::AIR.default_state();
        let rows: [(SpawnPlacementType, BlockStateId, BlockStateId); 3] = [
            (
                SpawnPlacementType::InWater,
                vanilla_blocks::WATER.default_state(),
                air,
            ),
            (
                SpawnPlacementType::InLava,
                vanilla_blocks::LAVA.default_state(),
                air,
            ),
            (
                SpawnPlacementType::OnGround,
                air,
                vanilla_blocks::STONE.default_state(),
            ),
        ];

        for (placement, at, below) in rows {
            let inside = TestLevel::default()
                .with_block(SPAWN_POS, at)
                .with_block(SPAWN_POS.below(), below);
            assert!(
                placement.is_spawn_position_ok(&inside, SPAWN_POS, zombie),
                "{placement:?} must accept this position inside the border, or the refusal proves nothing"
            );

            let outside = TestLevel::default()
                .with_block(SPAWN_POS, at)
                .with_block(SPAWN_POS.below(), below)
                .with_blocks_outside_world_border();
            assert!(
                !placement.is_spawn_position_ok(&outside, SPAWN_POS, zombie),
                "{placement:?} must consult the world border"
            );
        }
    }

    /// Each of `isValidEmptySpawnBlock`'s five terms refuses on its own.
    ///
    /// The five stand-ins are asserted to fail exactly one term apiece, so the predicate's single
    /// bool cannot hide a term that stopped being read.
    #[test]
    fn every_empty_spawn_block_term_refuses_alone() {
        init_vanilla_registry();
        init_behaviors();
        let zombie = &vanilla_entities::ZOMBIE;
        let level = TestLevel::default();
        let air = vanilla_blocks::AIR.default_state();

        assert!(
            failing_spawn_block_terms(&level, air).is_empty(),
            "air must fail nothing, or the refusals below prove nothing"
        );
        assert!(is_valid_empty_spawn_block(
            &level,
            SPAWN_POS,
            air,
            air.get_fluid_state(),
            zombie
        ));

        for (state, term) in [
            (
                vanilla_blocks::STONE.default_state(),
                "full collision shape",
            ),
            (
                vanilla_blocks::REDSTONE_TORCH.default_state(),
                "signal source",
            ),
            (vanilla_blocks::WATER.default_state(), "fluid"),
            (
                vanilla_blocks::RAIL.default_state(),
                "prevent_mob_spawning_inside",
            ),
            (vanilla_blocks::WITHER_ROSE.default_state(), "hazard"),
        ] {
            assert_eq!(
                failing_spawn_block_terms(&level, state),
                vec![term],
                "{} must stand in for exactly one term",
                state.get_block().key
            );
            assert!(
                !is_valid_empty_spawn_block(
                    &level,
                    SPAWN_POS,
                    state,
                    state.get_fluid_state(),
                    zombie
                ),
                "{} fails the {term} term and must refuse the spawn",
                state.get_block().key
            );
        }
    }

    /// The immunity table reaches the spawn predicate, not just the dismount path.
    ///
    /// A wither rose refuses a zombie's spawn and accepts a wither's, which is the one place
    /// [`is_block_dangerous`]'s per-type half is observable from here.
    #[test]
    fn the_hazard_term_is_per_entity_type() {
        init_vanilla_registry();
        init_behaviors();
        let level = TestLevel::default();
        let rose = vanilla_blocks::WITHER_ROSE.default_state();

        assert!(!is_valid_empty_spawn_block(
            &level,
            SPAWN_POS,
            rose,
            rose.get_fluid_state(),
            &vanilla_entities::ZOMBIE
        ));
        assert!(is_valid_empty_spawn_block(
            &level,
            SPAWN_POS,
            rose,
            rose.get_fluid_state(),
            &vanilla_entities::WITHER
        ));
    }

    /// `adjustSpawnPosition` steps down for `ON_GROUND` alone, and only through a walkable block.
    #[test]
    fn only_on_ground_adjusts_the_spawn_position() {
        init_vanilla_registry();
        init_behaviors();
        let walkable = TestLevel::default();
        assert_eq!(
            SpawnPlacementType::OnGround.adjust_spawn_position(&walkable, SPAWN_POS),
            SPAWN_POS.below()
        );

        let plant = TestLevel::default().with_block(
            SPAWN_POS.below(),
            vanilla_blocks::SHORT_GRASS.default_state(),
        );
        assert_eq!(
            SpawnPlacementType::OnGround.adjust_spawn_position(&plant, SPAWN_POS),
            SPAWN_POS.below(),
            "a candidate picked above a plant steps down into it"
        );

        let solid = TestLevel::default()
            .with_block(SPAWN_POS.below(), vanilla_blocks::STONE.default_state());
        assert_eq!(
            SpawnPlacementType::OnGround.adjust_spawn_position(&solid, SPAWN_POS),
            SPAWN_POS
        );

        for placement in [
            SpawnPlacementType::NoRestrictions,
            SpawnPlacementType::InWater,
            SpawnPlacementType::InLava,
        ] {
            assert_eq!(
                placement.adjust_spawn_position(&walkable, SPAWN_POS),
                SPAWN_POS,
                "{placement:?} takes vanilla's default, which returns the candidate untouched"
            );
        }
    }

    /// The free dispatcher routes through the registered row rather than answering for itself.
    #[test]
    fn spawn_position_dispatches_through_the_registered_row() {
        init_vanilla_registry();
        let lava = TestLevel::default().with_block(SPAWN_POS, vanilla_blocks::LAVA.default_state());

        assert!(is_spawn_position_ok(
            &vanilla_entities::STRIDER,
            &lava,
            SPAWN_POS
        ));
        assert!(
            !is_spawn_position_ok(&vanilla_entities::COD, &lava, SPAWN_POS),
            "the cod's IN_WATER row must refuse lava"
        );

        let outside = TestLevel::default().with_blocks_outside_world_border();
        assert!(
            is_spawn_position_ok(&vanilla_entities::ARMOR_STAND, &outside, SPAWN_POS),
            "an unregistered type falls back to NO_RESTRICTIONS, which reads nothing"
        );
    }
}
