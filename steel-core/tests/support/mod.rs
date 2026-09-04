use std::cell::{Cell, RefCell};
use std::slice;
use std::sync::{Arc, OnceLock};

use glam::DVec3;
use steel_registry::blocks::{BlockRef, block_state_ext::BlockStateExt};
use steel_registry::dimension_type::DimensionTypeRef;
use steel_registry::fluid::FluidRef;
use steel_registry::game_events::GameEventRef;
use steel_registry::sound_event::SoundEventRef;
use steel_registry::{
    init_vanilla_registry, vanilla_blocks, vanilla_dimension_types, vanilla_fluids,
};
use steel_utils::types::{Difficulty, GameType, UpdateFlags};
use steel_utils::{BlockPos, BlockStateId, Identifier};
use tokio::runtime::{Builder, Runtime};
use toml::map::Map;

use crate::chunk::Chunk;
use crate::chunk::chunk_holder::{ChunkHolder, TickingReadiness};
use crate::chunk::chunk_ticket_manager::ChunkTicketLevel;
use crate::chunk::light::LightLayer;
use crate::chunk::section::{ChunkSection, Sections};
use crate::chunk::status::ChunkStatus;
use crate::entity::Entity;
use crate::level_data::WorldGenerationSettings;
use crate::world::game_event::GameEventContext;
use crate::world::{
    LevelAccessor, LevelReader, ScheduledTickAccess, ServerLevelAccessor, World, WorldConfig,
    WorldStorageConfig,
};
use crate::worldgen::{ChunkGeneratorType, EmptyChunkGenerator};
use steel_utils::ChunkPos;

mod connection;
mod player;

pub(crate) use connection::TestConnection;
pub(crate) use player::{TestPlayerBuilder, test_runtime_config};

pub(crate) fn test_world() -> &'static Arc<World> {
    static WORLD: OnceLock<Arc<World>> = OnceLock::new();
    WORLD.get_or_init(|| create_test_world("test"))
}

pub(crate) fn fresh_test_world(key: &'static str) -> Arc<World> {
    create_test_world(key)
}

pub(crate) fn fresh_test_world_in_domain(domain: &'static str, key: &'static str) -> Arc<World> {
    create_test_world_with_key(Identifier::new_static(domain, key), Difficulty::Normal)
}

pub(crate) fn insert_ready_full_chunk(world: &Arc<World>, pos: ChunkPos) -> Arc<ChunkHolder> {
    insert_full_chunk(world, pos, true)
}

pub(crate) fn insert_unready_full_chunk(world: &Arc<World>, pos: ChunkPos) -> Arc<ChunkHolder> {
    insert_full_chunk(world, pos, false)
}

pub(crate) fn insert_full_chunk(
    world: &Arc<World>,
    pos: ChunkPos,
    block_ticking_ready: bool,
) -> Arc<ChunkHolder> {
    let min_y = world.get_min_y();
    let height = world.get_height();
    let sections = (0..height / 16)
        .map(|_| ChunkSection::new_empty())
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let proto = Chunk::new(
        Sections::from_owned(sections),
        pos,
        min_y,
        height,
        Arc::downgrade(world),
    );
    let _ = proto.promote_to_full();
    let holder = Arc::new(ChunkHolder::new(
        pos,
        ChunkTicketLevel::BLOCK_TICKING_CHUNK,
        Some(ChunkTicketLevel::BLOCK_TICKING_CHUNK),
        min_y,
        height,
    ));
    holder.insert_chunk(proto, ChunkStatus::Full);
    if block_ticking_ready {
        assert_eq!(
            holder.transition_ticking_readiness(TickingReadiness::BlockTicking),
            Some(TickingReadiness::Unready)
        );
    }
    let _ = world.chunk_map.chunks.insert_sync(pos, Arc::clone(&holder));
    world.on_entity_chunk_loaded(pos);
    world.update_entity_chunk_visibility(pos, holder.entity_visibility());
    world
        .chunk_map
        .activate_block_entities(slice::from_ref(&holder));
    world.chunk_map.rebuild_ticking_chunk_snapshot();
    holder
}

pub(crate) fn cross_world_damage_test_world() -> &'static Arc<World> {
    static WORLD: OnceLock<Arc<World>> = OnceLock::new();
    WORLD.get_or_init(|| {
        let world = create_test_world("test_cross_world_damage");
        world.level_data.write().set_game_time(100);
        world
    })
}

pub(crate) fn hard_damage_test_world() -> &'static Arc<World> {
    static WORLD: OnceLock<Arc<World>> = OnceLock::new();
    WORLD.get_or_init(|| create_test_world_with_difficulty("test_hard_damage", Difficulty::Hard))
}

pub(crate) fn world_border_projectile_test_world() -> &'static Arc<World> {
    static WORLD: OnceLock<Arc<World>> = OnceLock::new();
    WORLD.get_or_init(|| {
        let world = create_test_world("test_world_border_projectile");
        let result = world.set_world_border_size(10.0);
        assert!(
            result.is_ok(),
            "test world border should resize: {result:?}"
        );
        world
    })
}

struct TestWorldResources {
    runtime: Arc<Runtime>,
    generation_pool: Arc<rayon::ThreadPool>,
}

fn test_world_resources() -> &'static TestWorldResources {
    static RESOURCES: OnceLock<TestWorldResources> = OnceLock::new();
    RESOURCES.get_or_init(|| TestWorldResources {
        runtime: Arc::new(
            Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .expect("test world runtime should start"),
        ),
        generation_pool: Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .thread_name(|index| format!("steel-test-world-{index}"))
                .build()
                .expect("test world generation pool should start"),
        ),
    })
}

fn create_test_world(key: &'static str) -> Arc<World> {
    create_test_world_with_difficulty(key, Difficulty::Normal)
}

fn create_test_world_with_difficulty(key: &'static str, difficulty: Difficulty) -> Arc<World> {
    create_test_world_with_key(Identifier::vanilla_static(key), difficulty)
}

fn create_test_world_with_key(key: Identifier, difficulty: Difficulty) -> Arc<World> {
    init_vanilla_registry();
    let resources = test_world_resources();
    let generator = Arc::new(ChunkGeneratorType::Empty(EmptyChunkGenerator::new()));
    let generator_config = toml::Value::Table(Map::new());
    let generation_settings = WorldGenerationSettings::from_generator_config(
        Identifier::vanilla_static("empty"),
        &generator_config,
        vanilla_dimension_types::OVERWORLD.key.clone(),
        vanilla_dimension_types::OVERWORLD.min_y,
        vanilla_dimension_types::OVERWORLD.height,
    );

    resources
        .runtime
        .block_on(World::new_with_config(
            Arc::clone(&resources.runtime),
            key,
            &vanilla_dimension_types::OVERWORLD,
            0,
            WorldConfig {
                storage: WorldStorageConfig::RamOnly,
                level_data_path: None,
                generator,
                generation_settings,
                view_distance: 2,
                simulation_distance: 2,
                max_chained_neighbor_updates: 1_000_000,
                compression: None,
                is_flat: false,
                sea_level: 63,
                default_gamemode: GameType::Survival,
                difficulty,
            },
            Arc::clone(&resources.generation_pool),
        ))
        .expect("test world should initialize")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlacedBlockState {
    pub(crate) pos: BlockPos,
    pub(crate) state: BlockStateId,
    pub(crate) flags: UpdateFlags,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScheduledBlockTick {
    pub(crate) pos: BlockPos,
    pub(crate) block: BlockRef,
    pub(crate) delay: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScheduledFluidTick {
    pub(crate) pos: BlockPos,
    pub(crate) fluid: FluidRef,
    pub(crate) delay: i32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PlayedBlockSound {
    pub(crate) sound: SoundEventRef,
    pub(crate) pos: BlockPos,
    pub(crate) volume: f32,
    pub(crate) pitch: f32,
    pub(crate) exclude: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecordedGameEvent {
    pub(crate) event: GameEventRef,
    pub(crate) pos: BlockPos,
    pub(crate) source_entity_id: Option<i32>,
    pub(crate) affected_state: Option<BlockStateId>,
}

pub(crate) struct TestLevel {
    blocks: RefCell<Vec<(BlockPos, BlockStateId)>>,
    default_block_state: RefCell<Option<BlockStateId>>,
    raw_brightness: Cell<u8>,
    sky_brightness: Cell<u8>,
    block_brightness: Cell<u8>,
    difficulty: Cell<Difficulty>,
    within_world_border: Cell<bool>,
    min_y: Cell<i32>,
    height: Cell<i32>,
    fluid_tick_delay: Cell<i32>,
    pub(crate) placed_blocks: RefCell<Vec<PlacedBlockState>>,
    pub(crate) scheduled_block_ticks: RefCell<Vec<ScheduledBlockTick>>,
    pub(crate) scheduled_fluid_ticks: RefCell<Vec<ScheduledFluidTick>>,
    pub(crate) block_sounds: RefCell<Vec<PlayedBlockSound>>,
    pub(crate) game_events: RefCell<Vec<RecordedGameEvent>>,
}

impl Default for TestLevel {
    fn default() -> Self {
        Self {
            blocks: RefCell::new(Vec::new()),
            default_block_state: RefCell::new(None),
            raw_brightness: Cell::new(0),
            sky_brightness: Cell::new(0),
            block_brightness: Cell::new(0),
            difficulty: Cell::new(Difficulty::Normal),
            within_world_border: Cell::new(true),
            min_y: Cell::new(-64),
            height: Cell::new(384),
            fluid_tick_delay: Cell::new(5),
            placed_blocks: RefCell::new(Vec::new()),
            scheduled_block_ticks: RefCell::new(Vec::new()),
            scheduled_fluid_ticks: RefCell::new(Vec::new()),
            block_sounds: RefCell::new(Vec::new()),
            game_events: RefCell::new(Vec::new()),
        }
    }
}

impl TestLevel {
    pub(crate) fn with_default_block_state(self, state: BlockStateId) -> Self {
        *self.default_block_state.borrow_mut() = Some(state);
        self
    }

    pub(crate) fn with_block(self, pos: BlockPos, state: BlockStateId) -> Self {
        self.set_test_block(pos, state);
        self
    }

    pub(crate) fn with_raw_brightness(self, raw_brightness: u8) -> Self {
        self.raw_brightness.set(raw_brightness);
        self
    }

    /// Sets the stored light of one layer, which `ServerLevelAccessor::brightness` reads.
    ///
    /// This is independent of [`Self::with_raw_brightness`], because the two are separate readings
    /// in vanilla too: `getBrightness` is one layer's stored value and `getRawBrightness` is the
    /// maximum across both after sky darkening.
    pub(crate) fn with_layer_brightness(self, layer: LightLayer, brightness: u8) -> Self {
        match layer {
            LightLayer::Sky => self.sky_brightness.set(brightness),
            LightLayer::Block => self.block_brightness.set(brightness),
        }
        self
    }

    /// Sets the difficulty this level reports.
    pub(crate) fn with_difficulty(self, difficulty: Difficulty) -> Self {
        self.difficulty.set(difficulty);
        self
    }

    /// Puts every position outside the world border.
    ///
    /// The fixture has no border geometry, so containment is one flag rather than a shape. That is
    /// enough for the callers that only ask the question — vanilla's spawn placements consult the
    /// border before reading a single block, and a test wanting the refusal wants it everywhere.
    pub(crate) fn with_blocks_outside_world_border(self) -> Self {
        self.within_world_border.set(false);
        self
    }

    pub(crate) fn with_min_y(self, min_y: i32) -> Self {
        self.min_y.set(min_y);
        self
    }

    pub(crate) fn with_height(self, height: i32) -> Self {
        self.height.set(height);
        self
    }

    pub(crate) fn set_test_block(&self, pos: BlockPos, state: BlockStateId) {
        let mut blocks = self.blocks.borrow_mut();
        if let Some((_, existing)) = blocks.iter_mut().find(|(block_pos, _)| *block_pos == pos) {
            *existing = state;
        } else {
            blocks.push((pos, state));
        }
    }

    pub(crate) fn last_placed_state(&self) -> Option<BlockStateId> {
        self.placed_blocks
            .borrow()
            .last()
            .map(|placed| placed.state)
    }

    pub(crate) fn scheduled_water_tick(&self) -> bool {
        self.scheduled_fluid_ticks
            .borrow()
            .iter()
            .any(|tick| tick.fluid == &vanilla_fluids::WATER)
    }
}

impl LevelReader for TestLevel {
    fn get_block_state(&self, pos: BlockPos) -> BlockStateId {
        self.blocks
            .borrow()
            .iter()
            .rev()
            .find(|(block_pos, _)| *block_pos == pos)
            .map_or_else(
                || {
                    self.default_block_state
                        .borrow()
                        .unwrap_or_else(|| vanilla_blocks::AIR.default_state())
                },
                |(_, state)| *state,
            )
    }

    fn raw_brightness(&self, _pos: BlockPos, _sky_darkening: u8) -> u8 {
        self.raw_brightness.get()
    }

    fn min_y(&self) -> i32 {
        self.min_y.get()
    }

    fn height(&self) -> i32 {
        self.height.get()
    }
}

impl ScheduledTickAccess for TestLevel {
    fn fluid_tick_delay(&self, _fluid: FluidRef) -> i32 {
        self.fluid_tick_delay.get()
    }

    fn schedule_block_tick_default(&self, pos: BlockPos, block: BlockRef, delay: i32) -> bool {
        self.scheduled_block_ticks
            .borrow_mut()
            .push(ScheduledBlockTick { pos, block, delay });
        true
    }

    fn schedule_fluid_tick_default(&self, pos: BlockPos, fluid: FluidRef, delay: i32) -> bool {
        self.scheduled_fluid_ticks
            .borrow_mut()
            .push(ScheduledFluidTick { pos, fluid, delay });
        true
    }
}

impl LevelAccessor for TestLevel {
    fn set_block_state(&self, pos: BlockPos, state: BlockStateId, flags: UpdateFlags) -> bool {
        self.set_test_block(pos, state);
        self.placed_blocks
            .borrow_mut()
            .push(PlacedBlockState { pos, state, flags });
        true
    }

    fn destroy_block(&self, pos: BlockPos, _drop_items: bool) -> bool {
        if self.get_block_state(pos).is_air() {
            return false;
        }

        self.set_block_state(
            pos,
            vanilla_blocks::AIR.default_state(),
            UpdateFlags::UPDATE_ALL,
        )
    }

    fn play_block_sound(
        &self,
        sound: SoundEventRef,
        pos: BlockPos,
        volume: f32,
        pitch: f32,
        exclude: Option<i32>,
    ) {
        self.block_sounds.borrow_mut().push(PlayedBlockSound {
            sound,
            pos,
            volume,
            pitch,
            exclude,
        });
    }

    fn game_event(&self, event: GameEventRef, pos: BlockPos, context: &GameEventContext<'_>) {
        self.game_events.borrow_mut().push(RecordedGameEvent {
            event,
            pos,
            source_entity_id: context.source_entity().map(Entity::id),
            affected_state: context.affected_state(),
        });
    }
}

/// Lets spawn predicates run against the fixture rather than a whole [`World`].
///
/// Three of these are fixed rather than configurable, because no test needs to vary them and a
/// setter nothing calls is a dead-code error here: the weather is calm, the sky is not darkened, and
/// no player is nearby. The dimension is the overworld, so the monster light window is its uniform
/// 0..=7 and its block-light limit is 0.
impl ServerLevelAccessor for TestLevel {
    fn difficulty(&self) -> Difficulty {
        self.difficulty.get()
    }

    fn brightness(&self, layer: LightLayer, _pos: BlockPos) -> u8 {
        match layer {
            LightLayer::Sky => self.sky_brightness.get(),
            LightLayer::Block => self.block_brightness.get(),
        }
    }

    fn is_thundering(&self) -> bool {
        false
    }

    fn sky_darkening(&self) -> u8 {
        0
    }

    fn dimension_type(&self) -> DimensionTypeRef {
        &vanilla_dimension_types::OVERWORLD
    }

    fn sea_level(&self) -> i32 {
        63
    }

    fn is_block_within_world_border(&self, _pos: BlockPos) -> bool {
        self.within_world_border.get()
    }

    fn has_nearby_non_creative_player(&self, _position: DVec3, _range: f64) -> bool {
        false
    }
}
