//! Read-only world view shared by live worlds and world-generation regions.
//!
//! This mirrors vanilla's `LevelReader` role: block behavior such as
//! `canSurvive` should depend on the world-reading surface, not on the concrete
//! `World` type. `World` and `WorldGenRegion` both implement this trait.

use glam::DVec3;
use steel_registry::blocks::BlockRef;
use steel_registry::blocks::block_state_ext::BlockStateExt as _;
use steel_registry::blocks::properties::Direction;
use steel_registry::blocks::shapes::{SupportType, is_shape_full_block};
use steel_registry::blocks::spawn_rule::BlockSpawnRule;
use steel_registry::dimension_type::DimensionTypeRef;
use steel_registry::entity_type::EntityTypeRef;
use steel_registry::fluid::FluidRef;
use steel_registry::game_events::GameEventRef;
use steel_registry::sound_event::SoundEventRef;
use steel_utils::types::Difficulty;
use steel_utils::{BlockPos, BlockStateId, types::UpdateFlags};

use crate::block_entity::SharedBlockEntity;
use crate::chunk::light::LightLayer;
use crate::world::game_event::GameEventContext;

const VANILLA_HORIZONTAL_LIMIT: i32 = 30_000_000;

/// Read-only level access needed by block behavior and worldgen predicates.
pub trait LevelReader {
    /// Gets the block state at a position.
    fn get_block_state(&self, pos: BlockPos) -> BlockStateId;

    /// Gets the block entity at a position when this level surface supports it
    #[expect(
        unused_variables,
        reason = "default trait implementation ignores position"
    )]
    fn get_block_entity(&self, pos: BlockPos) -> Option<SharedBlockEntity> {
        None
    }

    /// Mirrors vanilla `BlockState.isFaceSturdy` with full-face support.
    fn is_face_sturdy(&self, state: BlockStateId, pos: BlockPos, direction: Direction) -> bool {
        self.is_face_sturdy_for(state, pos, direction, SupportType::Full)
    }

    /// Mirrors vanilla `BlockState.isFaceSturdy` for a specific support type.
    ///
    /// Lightweight and worldgen views default to extracted support shapes.
    /// Live views override this to dispatch through block behavior for dynamic
    /// world-dependent shapes.
    fn is_face_sturdy_for(
        &self,
        state: BlockStateId,
        pos: BlockPos,
        direction: Direction,
        support_type: SupportType,
    ) -> bool {
        state.is_face_sturdy_for_at(pos, direction, support_type)
    }

    /// Mirrors vanilla `BlockState.isCollisionShapeFullBlock`.
    ///
    /// Lightweight and worldgen views default to the extracted static collision shape, which is
    /// exactly what vanilla's block-state cache precomputes: it measures the shape against
    /// `EmptyBlockGetter` at the origin, and refuses to build at all for a block carrying both a
    /// collision shape and a position offset. Live views override this to dispatch through block
    /// behavior, which is vanilla's uncached path for a dynamic-shape block.
    #[expect(
        unused_variables,
        reason = "the extracted default answers from the state alone"
    )]
    fn is_collision_shape_full_block(&self, state: BlockStateId, pos: BlockPos) -> bool {
        is_shape_full_block(state.get_static_collision_shape())
    }

    /// Returns vanilla `BlockState.isValidSpawn`.
    ///
    /// This is the per-block half of a natural spawn check: whether *this* block, sitting one
    /// below the candidate position, will hold the given entity type. Vanilla stores it as a
    /// predicate on `BlockBehaviour.Properties`, defaulting to a sturdy upward face on a block
    /// emitting less than light level 14, with 34 registrations in `Blocks` replacing that
    /// default. [`BlockSpawnRule`] carries those replacements; only the default needs a level and
    /// a position, which is why it is evaluated here rather than there.
    fn is_valid_spawn(
        &self,
        state: BlockStateId,
        pos: BlockPos,
        entity_type: EntityTypeRef,
    ) -> bool {
        if let Some(allowed) = BlockSpawnRule::of(state.get_block()).allows(entity_type) {
            return allowed;
        }

        self.is_face_sturdy(state, pos, Direction::Up) && state.get_light_emission() < 14
    }

    /// Returns vanilla raw brightness at a position after sky darkening.
    fn raw_brightness(&self, pos: BlockPos, sky_darkening: u8) -> u8;

    /// Returns vanilla `BlockAndLightGetter.canSeeSky`.
    fn can_see_sky(&self, pos: BlockPos) -> bool {
        self.raw_brightness(pos, 0) >= 15
    }

    /// Returns this dimension's vanilla ambient light factor.
    fn ambient_light(&self) -> f32 {
        0.0
    }

    /// Returns the minimum build height.
    fn min_y(&self) -> i32;

    /// Returns the build height.
    fn height(&self) -> i32;

    /// Returns the exclusive maximum build height.
    fn max_y_exclusive(&self) -> i32 {
        self.min_y() + self.height()
    }

    /// Checks if a Y coordinate is outside build height.
    fn is_outside_build_height(&self, y: i32) -> bool {
        y < self.min_y() || y >= self.max_y_exclusive()
    }

    /// Returns vanilla `LevelReader.getMaxLocalRawBrightness`.
    fn max_local_raw_brightness(&self, pos: BlockPos, sky_darkening: u8) -> u8 {
        if pos.x() < -VANILLA_HORIZONTAL_LIMIT
            || pos.z() < -VANILLA_HORIZONTAL_LIMIT
            || pos.x() >= VANILLA_HORIZONTAL_LIMIT
            || pos.z() >= VANILLA_HORIZONTAL_LIMIT
        {
            return 15;
        }

        self.raw_brightness(pos, sky_darkening)
    }

    /// Returns vanilla `LevelReader.getLightLevelDependentMagicValue`.
    fn light_level_dependent_magic_value(&self, pos: BlockPos) -> f32 {
        let value = f32::from(self.max_local_raw_brightness(pos, 0)) / 15.0;
        let curved_value = value / value.mul_add(-3.0, 4.0);
        curved_value + self.ambient_light() * (1.0 - curved_value)
    }

    /// Returns vanilla `LevelReader.getPathfindingCostFromLightLevels`.
    fn pathfinding_cost_from_light_levels(&self, pos: BlockPos) -> f32 {
        self.light_level_dependent_magic_value(pos) - 0.5
    }
}

/// Level access needed by vanilla block `updateShape` logic.
///
/// Vanilla passes both `LevelReader` and `ScheduledTickAccess` to block shape updates.
/// Steel combines those surfaces so the same block behavior can run against a live
/// `World` and a `WorldGenRegion`.
pub trait ScheduledTickAccess: LevelReader {
    /// Returns the fluid tick delay in this level.
    fn fluid_tick_delay(&self, fluid: FluidRef) -> i32;

    /// Schedules a block tick using vanilla's default priority.
    fn schedule_block_tick_default(&self, pos: BlockPos, block: BlockRef, delay: i32) -> bool;

    /// Returns whether a tick is already scheduled for the same `(pos, block)`.
    #[expect(
        unused_variables,
        reason = "most test/worldgen level surfaces do not track scheduled tick presence"
    )]
    fn has_scheduled_block_tick(&self, pos: BlockPos, block: BlockRef) -> bool {
        false
    }

    /// Returns whether the same `(pos, block)` was selected for this tick and has not started.
    #[expect(
        unused_variables,
        reason = "worldgen and most test level surfaces do not execute scheduled tick batches"
    )]
    fn will_tick_block_this_tick(&self, pos: BlockPos, block: BlockRef) -> bool {
        false
    }

    /// Schedules a fluid tick using vanilla's default priority.
    fn schedule_fluid_tick_default(&self, pos: BlockPos, fluid: FluidRef, delay: i32) -> bool;

    /// Returns whether the same `(pos, fluid)` was selected for this tick and has not started.
    #[expect(
        unused_variables,
        reason = "worldgen and most test level surfaces do not execute scheduled tick batches"
    )]
    fn will_tick_fluid_this_tick(&self, pos: BlockPos, fluid: FluidRef) -> bool {
        false
    }
}

/// Mutable level access needed by vanilla `LevelAccessor` block hooks.
pub trait LevelAccessor: ScheduledTickAccess {
    /// Sets a block state with vanilla update flags.
    fn set_block_state(&self, pos: BlockPos, state: BlockStateId, flags: UpdateFlags) -> bool;

    /// Destroys a block and optionally drops its resources.
    fn destroy_block(&self, pos: BlockPos, drop_items: bool) -> bool;

    /// Plays a block sound when this level surface supports runtime side effects.
    #[expect(
        unused_variables,
        reason = "worldgen and test level surfaces do not emit sounds"
    )]
    fn play_block_sound(
        &self,
        sound: SoundEventRef,
        pos: BlockPos,
        volume: f32,
        pitch: f32,
        exclude: Option<i32>,
    ) {
    }

    /// Dispatches a game event when this level surface supports runtime listeners.
    #[expect(
        unused_variables,
        reason = "worldgen and test level surfaces do not emit game events"
    )]
    fn game_event(&self, event: GameEventRef, pos: BlockPos, context: &GameEventContext<'_>) {}
}

/// Level access needed by vanilla spawn predicates.
///
/// This is vanilla's `ServerLevelAccessor`, the surface `SpawnPlacements.checkSpawnRules` takes.
/// Vanilla draws the line in the same place and for the same reason: a spawn predicate reads
/// world-wide state a plain block-reading view has no business knowing — the difficulty, the
/// weather, the per-layer light, the dimension's spawn light window, the sea level and the
/// players. Vanilla reaches most of it through `ServerLevelAccessor.getLevel()`, which returns
/// the real `ServerLevel` even when the caller is a worldgen region.
///
/// Outside tests, only [`crate::world::World`] and [`crate::worldgen::region::WorldGenRegion`]
/// implement this. Both must, because `NaturalSpawner.spawnMobsForChunkGeneration` calls
/// `checkSpawnRules` with a region rather than a level. Keeping these methods off [`LevelReader`]
/// is deliberate: that trait has a dozen minimal test and AI fixtures as impls, and defaulted
/// world-state readers would hand every one of them a plausible wrong answer.
pub trait ServerLevelAccessor: LevelAccessor {
    /// Returns vanilla `LevelAccessor.getDifficulty`.
    fn difficulty(&self) -> Difficulty;

    /// Returns vanilla `BlockAndTintGetter.getBrightness` for one light layer.
    ///
    /// This is the raw stored light of that single layer, unlike
    /// [`LevelReader::raw_brightness`], which is vanilla `getRawBrightness` and takes the maximum
    /// across both layers after sky darkening.
    fn brightness(&self, layer: LightLayer, pos: BlockPos) -> u8;

    /// Returns vanilla `Level.isThundering`.
    fn is_thundering(&self) -> bool;

    /// Returns vanilla `Level.getSkyDarken`.
    fn sky_darkening(&self) -> u8;

    /// Returns vanilla `LevelReader.dimensionType`.
    fn dimension_type(&self) -> DimensionTypeRef;

    /// Returns vanilla `LevelReader.getSeaLevel`.
    fn sea_level(&self) -> i32;

    /// Returns vanilla `WorldBorder.isWithinBounds(BlockPos)` for this level's border.
    ///
    /// Vanilla reaches the border through `LevelReader.getWorldBorder()`, which every reading view
    /// answers. Steel keeps it here for the same reason the rest of this trait is here: the border
    /// is world-wide state, and a defaulted reader on [`LevelReader`] would hand each of that
    /// trait's minimal fixtures a plausible wrong answer. Three of vanilla's four placement types
    /// gate on it, so a fixture answering `true` by accident would let a spawn through that the
    /// live world refuses.
    fn is_block_within_world_border(&self, pos: BlockPos) -> bool;

    /// Returns whether vanilla `EntityGetter.getNearestPlayer(x, y, z, range, true)` finds anyone.
    ///
    /// That `true` selects `EntitySelector.NO_CREATIVE_OR_SPECTATOR`, so creative-mode and
    /// spectating players are invisible to the query. Spawn predicates only ever compare the
    /// result against `null`, so this reports presence rather than returning the player.
    fn has_nearby_non_creative_player(&self, position: DVec3, range: f64) -> bool;

    /// Returns vanilla `LevelReader.getMaxLocalRawBrightness(pos)`, the one-argument form.
    ///
    /// Vanilla's overload passes the level's current sky darkening, which a plain reading view
    /// cannot supply, so the two-argument form lives on [`LevelReader`] and this one here.
    fn max_local_raw_brightness_now(&self, pos: BlockPos) -> u8 {
        self.max_local_raw_brightness(pos, self.sky_darkening())
    }
}

#[cfg(test)]
mod tests {
    use steel_registry::{init_vanilla_registry, vanilla_blocks, vanilla_entities};

    use super::*;

    struct TestLevel {
        raw_brightness: u8,
        ambient_light: f32,
    }

    impl LevelReader for TestLevel {
        fn get_block_state(&self, _pos: BlockPos) -> BlockStateId {
            BlockStateId(0)
        }

        fn raw_brightness(&self, _pos: BlockPos, _sky_darkening: u8) -> u8 {
            self.raw_brightness
        }

        fn ambient_light(&self) -> f32 {
            self.ambient_light
        }

        fn min_y(&self) -> i32 {
            -64
        }

        fn height(&self) -> i32 {
            384
        }
    }

    fn assert_f32_close(left: f32, right: f32) {
        assert!(
            (left - right).abs() < 0.000_001,
            "left={left}, right={right}"
        );
    }

    #[test]
    fn pathfinding_cost_uses_vanilla_curved_light_value() {
        let level = TestLevel {
            raw_brightness: 6,
            ambient_light: 0.0,
        };

        assert_f32_close(
            level.pathfinding_cost_from_light_levels(BlockPos::ZERO),
            -0.357_142_87,
        );
    }

    #[test]
    fn pathfinding_cost_lerps_toward_full_light_with_ambient_light() {
        let level = TestLevel {
            raw_brightness: 6,
            ambient_light: 0.2,
        };

        assert_f32_close(
            level.pathfinding_cost_from_light_levels(BlockPos::ZERO),
            -0.185_714_3,
        );
    }

    #[test]
    fn max_local_raw_brightness_matches_vanilla_horizontal_limit() {
        let level = TestLevel {
            raw_brightness: 0,
            ambient_light: 0.0,
        };

        assert_eq!(
            level.max_local_raw_brightness(BlockPos::new(29_999_999, 64, 0), 0),
            0
        );
        assert_eq!(
            level.max_local_raw_brightness(BlockPos::new(30_000_000, 64, 0), 0),
            15
        );
    }

    #[test]
    fn can_see_sky_uses_vanilla_sky_light_threshold() {
        assert!(
            TestLevel {
                raw_brightness: 15,
                ambient_light: 0.0,
            }
            .can_see_sky(BlockPos::ZERO)
        );
        assert!(
            !TestLevel {
                raw_brightness: 14,
                ambient_light: 0.0,
            }
            .can_see_sky(BlockPos::ZERO)
        );
    }

    /// A level with no light of its own, so `is_valid_spawn` reads only the block it is handed.
    fn unlit_level() -> TestLevel {
        init_vanilla_registry();
        TestLevel {
            raw_brightness: 0,
            ambient_light: 0.0,
        }
    }

    /// A plain full block that emits nothing satisfies vanilla's default predicate.
    #[test]
    fn default_spawn_predicate_accepts_a_sturdy_unlit_block() {
        let level = unlit_level();

        assert!(level.is_valid_spawn(
            vanilla_blocks::STONE.default_state(),
            BlockPos::ZERO,
            &vanilla_entities::ZOMBIE,
        ));
    }

    /// Light emission of 14 or more fails the default predicate on an otherwise sturdy block.
    #[test]
    fn default_spawn_predicate_rejects_a_bright_block() {
        let level = unlit_level();
        let glowstone = vanilla_blocks::GLOWSTONE.default_state();

        assert!(
            level.is_face_sturdy(glowstone, BlockPos::ZERO, Direction::Up),
            "glowstone must hold the sturdy half of the default predicate, or this test proves nothing"
        );
        assert!(glowstone.get_light_emission() >= 14);
        assert!(!level.is_valid_spawn(glowstone, BlockPos::ZERO, &vanilla_entities::ZOMBIE));
    }

    /// A tabled rule beats the default: bedrock passes the default and still refuses every spawn.
    #[test]
    fn tabled_rules_override_the_default_predicate() {
        let level = unlit_level();
        let bedrock = vanilla_blocks::BEDROCK.default_state();

        assert!(
            level.is_face_sturdy(bedrock, BlockPos::ZERO, Direction::Up)
                && bedrock.get_light_emission() < 14,
            "bedrock must satisfy the default predicate, or this test proves nothing"
        );
        assert!(!level.is_valid_spawn(bedrock, BlockPos::ZERO, &vanilla_entities::ZOMBIE));
    }

    /// The entity-typed rules read the spawning type rather than the block alone.
    ///
    /// Leaves and ice bracket the two directions a rule can move the answer. Vanilla
    /// `LeavesBlock.getBlockSupportShape` returns an empty shape, so the default predicate refuses
    /// *every* mob on leaves and `ocelotOrParrot` is what lets two of them through; ice and the
    /// magma block both satisfy the default, so their rules exist only to turn it down.
    #[test]
    fn entity_typed_rules_select_on_the_spawning_type() {
        let level = unlit_level();
        let pos = BlockPos::ZERO;

        let leaves = vanilla_blocks::OAK_LEAVES.default_state();
        assert!(
            !level.is_face_sturdy(leaves, pos, Direction::Up),
            "leaves must fail the default predicate, or this test proves nothing"
        );
        assert!(level.is_valid_spawn(leaves, pos, &vanilla_entities::OCELOT));
        assert!(level.is_valid_spawn(leaves, pos, &vanilla_entities::PARROT));
        assert!(!level.is_valid_spawn(leaves, pos, &vanilla_entities::ZOMBIE));

        for state in [
            vanilla_blocks::ICE.default_state(),
            vanilla_blocks::MAGMA_BLOCK.default_state(),
        ] {
            assert!(
                level.is_face_sturdy(state, pos, Direction::Up) && state.get_light_emission() < 14,
                "{} must satisfy the default predicate, or this test proves nothing",
                state.get_block().key
            );
        }

        let ice = vanilla_blocks::ICE.default_state();
        assert!(level.is_valid_spawn(ice, pos, &vanilla_entities::POLAR_BEAR));
        assert!(!level.is_valid_spawn(ice, pos, &vanilla_entities::COW));

        let magma = vanilla_blocks::MAGMA_BLOCK.default_state();
        assert!(level.is_valid_spawn(magma, pos, &vanilla_entities::BLAZE));
        assert!(!level.is_valid_spawn(magma, pos, &vanilla_entities::COW));
    }
}
