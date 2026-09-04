//! Chunk-generation mob spawning, vanilla's `ChunkGenerator.spawnOriginalMobs`.
//!
//! This is the stage that makes a freshly generated world already hold animals: the per-tick
//! spawn loop only reaches chunks a player is near, and its `creature` category runs on a
//! 400-tick stride, so without this stage a new world's herds would trickle in over minutes
//! instead of being there when the chunk first appears.

use std::sync::Arc;

use steel_registry::REGISTRY;
use steel_registry::RegistryExt as _;
use steel_utils::random::Random as _;

use crate::chunk::{
    chunk_generation_task::StaticCache2D, chunk_holder::ChunkHolder, chunk_pyramid::ChunkStep,
};
use crate::world::natural_spawner::spawn_mobs_for_chunk_generation;
use crate::worldgen::generator::ChunkGenerator as _;
use crate::worldgen::generator::context::WorldGenContext;
use crate::worldgen::region::WorldGenRegion;

pub(crate) fn generate(
    context: Arc<WorldGenContext>,
    step: &ChunkStep,
    cache: &Arc<StaticCache2D<Arc<ChunkHolder>>>,
    holder: Arc<ChunkHolder>,
) {
    let center = holder.get_pos();
    let world_seed = context.world().seed();
    let region_random = context
        .generator
        .create_worldgen_region_random(world_seed, center);
    let mut region = WorldGenRegion::new(context.as_ref(), step, cache, center, region_random);

    // Vanilla reads the biome at the column's top rather than at the spawn positions, so one
    // chunk draws its whole generation-time population from a single biome.
    let top_y = region.min_y() + region.height() - 1;
    let biome_id = region.noise_biome_id(center.0.x * 4, top_y >> 2, center.0.y * 4);
    let Some(biome) = REGISTRY.biomes.by_id(usize::from(biome_id)) else {
        return;
    };

    let mut random = context
        .generator
        .create_worldgen_region_random(world_seed, center);
    // Vanilla seeds this from a fresh unique seed rather than from the world seed, so two worlds
    // with the same terrain do not get identical herds. Steel's region random is already
    // per-chunk, so one extra draw is enough to decorrelate it from the decoration stream.
    let _ = random.next_i32();

    spawn_mobs_for_chunk_generation(&region, biome, center, &mut random);
    let _ = region.random_mut();
}
