//! Vanilla's natural-spawn candidate chunks and the 128-block player filter over them.
//!
//! The numbers here are worked from vanilla's own expressions rather than read back from Steel:
//! the candidate square is `FixedPlayerDistanceChunkTracker(8)`'s 17x17 Chebyshev neighborhood,
//! and every block distance is measured from the chunk's center column, `chunkCoord * 16 + 8`,
//! against `16384.0`.

use super::super::spawning_chunks::PlayersNearby;
use super::*;
use crate::test_support::insert_unready_full_chunk;
use glam::DVec3;
use steel_utils::types::GameType;

/// The center of a chunk's block column, where vanilla measures spawn distance from.
fn chunk_center(chunk_x: i32, chunk_z: i32) -> DVec3 {
    DVec3::new(
        f64::from(chunk_x) * 16.0 + 8.0,
        64.0,
        f64::from(chunk_z) * 16.0 + 8.0,
    )
}

/// Builds a player standing at `position` and joins it to `world`.
///
/// The position is set before the join, because a joined player's moves go through the world
/// entity manager and these tests only need the coordinate the spawn sample reads.
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
fn spawn_candidates_are_the_seventeen_by_seventeen_square_around_a_player() {
    init_vanilla_registry();
    init_behaviors();
    let world = fresh_test_world("spawn_candidate_square");
    let player = joined_player(&world, "SpawnCandidate", 1, chunk_center(0, 0));

    let players = world.chunk_map.spawning_players();

    assert!(players.is_spawn_candidate(ChunkPos::new(8, 8)));
    assert!(players.is_spawn_candidate(ChunkPos::new(-8, 8)));
    assert!(!players.is_spawn_candidate(ChunkPos::new(9, 0)));
    assert!(!players.is_spawn_candidate(ChunkPos::new(0, -9)));
    assert_eq!(players.natural_spawn_chunk_count(), 17 * 17);

    world.remove_player_for_world_change(&player);
}

#[test]
fn spawn_candidate_count_unions_overlapping_player_squares() {
    init_vanilla_registry();
    init_behaviors();
    let world = fresh_test_world("spawn_candidate_union");
    let first = joined_player(&world, "SpawnUnionFirst", 1, chunk_center(0, 0));
    let second = joined_player(&world, "SpawnUnionSecond", 2, chunk_center(1, 0));

    // Two occupied chunks one apart widen the square by a single column rather than doubling it.
    assert_eq!(
        world
            .chunk_map
            .spawning_players()
            .natural_spawn_chunk_count(),
        18 * 17
    );

    world.remove_player_for_world_change(&first);
    world.remove_player_for_world_change(&second);
}

#[test]
fn the_player_filter_measures_a_cylinder_from_the_chunk_center() {
    init_vanilla_registry();
    init_behaviors();
    let world = fresh_test_world("spawn_player_filter");
    let player = joined_player(&world, "SpawnFilter", 1, chunk_center(0, 0));

    let players = world.chunk_map.spawning_players();

    // Seven chunks out on one axis is 112 blocks center to center, and 12544 < 16384.
    assert!(players.any_player_close_enough_for_spawning_internal(ChunkPos::new(7, 0)));
    // Eight chunks out is exactly 128 blocks, and vanilla's comparison is strict.
    assert!(!players.any_player_close_enough_for_spawning_internal(ChunkPos::new(8, 0)));
    // Five chunks out on both axes is 80 blocks each, and 12800 < 16384.
    assert!(players.any_player_close_enough_for_spawning_internal(ChunkPos::new(5, 5)));
    // Six chunks out on both axes is 96 blocks each: 18432, so a candidate chunk can still fail.
    assert!(players.is_spawn_candidate(ChunkPos::new(6, 6)));
    assert!(!players.any_player_close_enough_for_spawning_internal(ChunkPos::new(6, 6)));

    // The world-level entry point samples the players itself and agrees with the sample above.
    assert!(
        world
            .chunk_map
            .any_player_close_enough_for_spawning(ChunkPos::new(7, 0))
    );
    assert!(
        !world
            .chunk_map
            .any_player_close_enough_for_spawning(ChunkPos::new(8, 0))
    );

    world.remove_player_for_world_change(&player);
}

#[test]
fn the_chunk_distance_pre_filter_only_decides_outside_the_undecided_ring() {
    init_vanilla_registry();
    init_behaviors();
    let world = fresh_test_world("spawn_pre_filter");
    let player = joined_player(&world, "SpawnPreFilter", 1, chunk_center(0, 0));

    let players = world.chunk_map.spawning_players();

    assert_eq!(
        players.has_players_nearby(ChunkPos::new(5, 5)),
        PlayersNearby::Yes
    );
    assert_eq!(
        players.has_players_nearby(ChunkPos::new(6, 0)),
        PlayersNearby::Undecided
    );
    assert_eq!(
        players.has_players_nearby(ChunkPos::new(8, 8)),
        PlayersNearby::Undecided
    );
    assert_eq!(
        players.has_players_nearby(ChunkPos::new(9, 0)),
        PlayersNearby::No
    );

    world.remove_player_for_world_change(&player);
}

#[test]
fn a_spectator_counts_toward_the_candidate_set_but_never_enables_spawning() {
    init_vanilla_registry();
    init_behaviors();
    let world = fresh_test_world("spawn_spectator");
    let player = joined_player(&world, "SpawnSpectator", 1, chunk_center(0, 0));
    player.restore_game_modes(GameType::Spectator, Some(GameType::Survival));

    let players = world.chunk_map.spawning_players();
    let center = ChunkPos::new(0, 0);

    // `spectators_generate_chunks` defaults to true, so vanilla's `skipPlayer` keeps them.
    assert_eq!(players.natural_spawn_chunk_count(), 17 * 17);
    assert!(!players.any_player_close_enough_for_spawning_internal(center));
    assert!(players.players_close_for_spawning(center).is_empty());
    // Vanilla's own quirk: the public query takes the pre-filter's `TRUE` at face value.
    assert!(players.any_player_close_enough_for_spawning(center));

    world.remove_player_for_world_change(&player);
}

#[test]
fn players_close_for_spawning_lists_only_the_players_in_range() {
    init_vanilla_registry();
    init_behaviors();
    let world = fresh_test_world("spawn_close_players");
    let near = joined_player(&world, "SpawnNear", 1, chunk_center(0, 0));
    let far = joined_player(&world, "SpawnFar", 2, chunk_center(20, 0));

    let players = world.chunk_map.spawning_players();

    let at_near = players.players_close_for_spawning(ChunkPos::new(0, 0));
    assert_eq!(at_near.len(), 1);
    assert!(Arc::ptr_eq(&at_near[0], &near));

    let at_far = players.players_close_for_spawning(ChunkPos::new(20, 0));
    assert_eq!(at_far.len(), 1);
    assert!(Arc::ptr_eq(&at_far[0], &far));

    // Ten chunks from both occupied chunks, so the pre-filter answers without any block test.
    assert!(
        players
            .players_close_for_spawning(ChunkPos::new(10, 0))
            .is_empty()
    );

    world.remove_player_for_world_change(&near);
    world.remove_player_for_world_change(&far);
}

#[test]
fn collect_spawning_chunks_keeps_only_block_ticking_chunks_near_a_player() {
    init_vanilla_registry();
    init_behaviors();
    let world = fresh_test_world("collect_spawning_chunks");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    // A candidate well inside 128 blocks, but never block-ticking.
    insert_unready_full_chunk(&world, ChunkPos::new(1, 0));
    // Block-ticking and a candidate, but 96 blocks out on both axes.
    insert_ready_full_chunk(&world, ChunkPos::new(6, 6));
    // Block-ticking, but twenty chunks out, so not a candidate at all.
    insert_ready_full_chunk(&world, ChunkPos::new(20, 0));
    let player = joined_player(&world, "SpawnCollector", 1, chunk_center(0, 0));

    let players = world.chunk_map.spawning_players();
    let collected = world
        .chunk_map
        .collect_spawning_chunks(&players)
        .iter()
        .map(|holder| holder.get_pos())
        .collect::<Vec<_>>();

    assert_eq!(collected, [ChunkPos::new(0, 0)]);

    world.remove_player_for_world_change(&player);
}

#[test]
fn inhabited_time_accrues_the_elapsed_interval_rather_than_one_tick() {
    init_vanilla_registry();
    init_behaviors();
    let world = fresh_test_world("inhabited_time_interval");
    let pos = ChunkPos::new(0, 0);
    insert_ready_full_chunk(&world, pos);
    let player = joined_player(&world, "InhabitedInterval", 1, chunk_center(0, 0));

    let players = world.chunk_map.spawning_players();
    let collected = world.chunk_map.collect_spawning_chunks(&players);
    assert_eq!(collected.len(), 1, "the player's own chunk should collect");

    // First pass from game time 0 credits nothing: the clock starts there, so the delta is zero.
    world.chunk_map.accrue_inhabited_time(0, &collected);
    assert_eq!(chunk_inhabited_time(&world, pos), 0);

    // A single tick credits one, and a twenty-tick jump credits twenty rather than one -- this is
    // the whole reason vanilla passes a delta instead of incrementing by a constant.
    world.chunk_map.accrue_inhabited_time(1, &collected);
    assert_eq!(chunk_inhabited_time(&world, pos), 1);
    world.chunk_map.accrue_inhabited_time(21, &collected);
    assert_eq!(chunk_inhabited_time(&world, pos), 21);

    world.remove_player_for_world_change(&player);
}

#[test]
fn the_inhabited_clock_advances_while_no_chunk_is_eligible() {
    init_vanilla_registry();
    init_behaviors();
    let world = fresh_test_world("inhabited_time_clock");
    let pos = ChunkPos::new(0, 0);
    insert_ready_full_chunk(&world, pos);
    let player = joined_player(&world, "InhabitedClock", 1, chunk_center(0, 0));

    let players = world.chunk_map.spawning_players();
    let collected = world.chunk_map.collect_spawning_chunks(&players);

    // Nobody is near for a hundred ticks. Vanilla still moves lastInhabitedUpdate, because it
    // does so in tick() before deciding whether any chunk is worth ticking.
    world.chunk_map.accrue_inhabited_time(0, &[]);
    world.chunk_map.accrue_inhabited_time(100, &[]);
    assert_eq!(chunk_inhabited_time(&world, pos), 0);

    // So the first eligible pass afterwards charges only its own tick, not the idle century.
    world.chunk_map.accrue_inhabited_time(101, &collected);
    assert_eq!(chunk_inhabited_time(&world, pos), 1);

    world.remove_player_for_world_change(&player);
}

#[test]
fn accruing_inhabited_time_leaves_the_chunk_clean() {
    init_vanilla_registry();
    init_behaviors();
    let world = fresh_test_world("inhabited_time_clean");
    let pos = ChunkPos::new(0, 0);
    insert_ready_full_chunk(&world, pos);
    let player = joined_player(&world, "InhabitedClean", 1, chunk_center(0, 0));

    let players = world.chunk_map.spawning_players();
    let collected = world.chunk_map.collect_spawning_chunks(&players);
    let Some(holder) = world.chunk_map.active_full_chunk_holder(pos) else {
        panic!("the inserted chunk should have a holder");
    };
    let Some(chunk) = holder.try_chunk(ChunkStatus::Full) else {
        panic!("the inserted chunk should be Full");
    };
    chunk.clear_dirty();

    world.chunk_map.accrue_inhabited_time(0, &collected);
    world.chunk_map.accrue_inhabited_time(40, &collected);

    assert_eq!(chunk.inhabited_time(), 40);
    // Vanilla's incrementInhabitedTime never calls markUnsaved, so time alone does not schedule
    // a save. Marking dirty here would make every player-occupied chunk rewrite itself forever.
    assert!(!chunk.is_dirty());

    world.remove_player_for_world_change(&player);
}

/// Reads one chunk's accrued `inhabitedTime` through its holder.
fn chunk_inhabited_time(world: &Arc<World>, pos: ChunkPos) -> i64 {
    let Some(holder) = world.chunk_map.active_full_chunk_holder(pos) else {
        panic!("chunk {pos:?} should have a holder");
    };
    let Some(chunk) = holder.try_chunk(ChunkStatus::Full) else {
        panic!("chunk {pos:?} should be Full");
    };
    chunk.inhabited_time()
}
