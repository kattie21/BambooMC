//! Vanilla's `StructureManager` reads, the half natural spawning needs.
//!
//! Ports the query side of `net.minecraft.world.level.StructureManager`: which structures reference
//! the chunk a block sits in, which of their starts are valid, and whether the block lies inside a
//! start's own bounding box or inside one of its pieces. The write side — `setStartForStructure`,
//! `addReferenceForStructure`, `checkStructurePresence` — belongs to chunk generation and already
//! lives in `steel_worldgen`, so nothing here mutates a chunk.
//!
//! Vanilla holds one `StructureManager` per level and reaches chunks through
//! `LevelAccessor.getChunk(x, z, status)`. Steel has no such object, so these are free functions
//! over a [`World`]: the manager carries no state a read needs beyond the level itself.
//!
//! One deliberate difference. Vanilla's reads *generate* a chunk that has not reached the requested
//! status, while [`crate::ChunkMap::with_chunk_at_status`] answers `None`. Every caller here is
//! asking about a block in a loaded ticking chunk, whose neighborhood passed both structure
//! statuses long ago, and generating a chunk synchronously inside the spawn loop is precisely what
//! Steel's architecture forbids — so the difference is unreachable where it would matter, and the
//! right answer where it is not.

use steel_registry::structure::StructureRef;
use steel_worldgen::structure::StructureStart;

use super::{BlockPos, ChunkPos, ChunkStatus, Identifier, REGISTRY, RegistryExt, World};

/// Vanilla `StructureManager.getAllStructuresAt`, copied out of the chunk rather than borrowed.
///
/// Vanilla hands back an unmodifiable view of the live chunk map and reads it while nothing else can
/// touch the chunk. Steel would have to hold that chunk's `structure_references` read guard for the
/// whole scan, and the scan then takes *other* chunks' `structure_starts` guards — a lock order this
/// module declines to establish. Copying first costs one allocation per referencing structure, and
/// none at all in the case that dominates every spawn tick: a chunk with no references yields an
/// empty `Vec` without allocating.
///
/// A referenced id the registry cannot resolve is skipped. Vanilla cannot reach that state because
/// its per-chunk map is keyed by resolved `Structure` objects and deserialization drops ids it does
/// not know, so skipping here is the same outcome one layer further down.
pub(crate) fn all_structures_at(
    world: &World,
    pos: BlockPos,
) -> Vec<(StructureRef, Vec<ChunkPos>)> {
    world
        .chunk_map
        .with_chunk_at_status(
            ChunkPos::from_block_pos(pos),
            ChunkStatus::StructureReferences,
            |chunk| {
                chunk
                    .structure_references()
                    .iter()
                    .filter_map(|(id, origins)| {
                        let structure = REGISTRY.structures.by_key(id)?;
                        Some((structure, origins.iter().copied().collect()))
                    })
                    .collect()
            },
        )
        .unwrap_or_default()
}

/// Vanilla `ChunkAccess.getReferencesForStructure`, reached the way the manager reaches it.
///
/// The one-key form of [`all_structures_at`], for the fortress check that already knows which
/// structure it is asking about. An absent key answers with vanilla's `EMPTY_REFERENCE_SET`, which
/// an empty `Vec` is.
pub(crate) fn references_for_structure(
    world: &World,
    pos: BlockPos,
    structure: &Identifier,
) -> Vec<ChunkPos> {
    world
        .chunk_map
        .with_chunk_at_status(
            ChunkPos::from_block_pos(pos),
            ChunkStatus::StructureReferences,
            |chunk| {
                chunk
                    .structure_references()
                    .get(structure)
                    .map_or_else(Vec::new, |origins| origins.iter().copied().collect())
            },
        )
        .unwrap_or_default()
}

/// Vanilla `StructureManager.fillStartsForStructure`, folded into the boolean its callers want.
///
/// Vanilla pushes every valid start into a consumer. Neither caller in the spawning path keeps the
/// starts — `ChunkGenerator.getMobsAt` folds them into a flag and `getStructureAt` returns on the
/// first box that contains the position — so this answers the flag directly and allocates nothing.
/// `getMobsAt` does finish its loop after the flag is set where this stops early; with a pure
/// predicate, and both predicates below read only the position and the start, the answer is the same.
///
/// Validity is vanilla's `StructureStart.isValid()`, `!pieceContainer.isEmpty()`. Steel's start
/// documents `bounding_box: None` exactly when `pieces` is empty, so the two tests agree by
/// construction and [`structure_start_contains`] needs no separate guard.
pub(crate) fn any_start_matching(
    world: &World,
    structure: &Identifier,
    origins: &[ChunkPos],
    predicate: impl Fn(&StructureStart) -> bool,
) -> bool {
    origins.iter().any(|origin| {
        world
            .chunk_map
            .with_chunk_at_status(*origin, ChunkStatus::StructureStarts, |chunk| {
                chunk
                    .structure_starts()
                    .get(structure)
                    .is_some_and(|start| !start.pieces.is_empty() && predicate(start))
            })
            .unwrap_or(false)
    })
}

/// Vanilla `StructureManager.structureHasPieceAt`, the `piece` spawn-override box.
///
/// Tests the piece boxes themselves rather than their union, so the gaps between a structure's
/// pieces fall outside the override — which is how a swamp hut spawns its witch only inside the hut.
pub(crate) fn structure_has_piece_at(pos: BlockPos, start: &StructureStart) -> bool {
    start
        .pieces
        .iter()
        .any(|piece| piece.bounding_box.contains_blockpos(pos))
}

/// Vanilla `start.getBoundingBox().isInside(pos)`, the `full` spawn-override box.
///
/// The union of every piece box, inflated by the structure's terrain adaptation, so this accepts
/// positions no piece covers. An ocean monument overriding with `full` therefore claims the water
/// around its walls, which is the point of the mode.
pub(crate) fn structure_start_contains(pos: BlockPos, start: &StructureStart) -> bool {
    start
        .bounding_box
        .is_some_and(|bounding_box| bounding_box.contains_blockpos(pos))
}

/// Vanilla `StructureManager.getStructureAt(pos, structure).isValid()`.
///
/// Vanilla returns the matching start and falls back to `StructureStart.INVALID_START`; its caller
/// in the spawning path only asks `isValid()`, so this answers that and there is no invalid sentinel
/// to model.
pub(crate) fn has_structure_at(world: &World, pos: BlockPos, structure: &Identifier) -> bool {
    let origins = references_for_structure(world, pos, structure);

    any_start_matching(world, structure, &origins, |start| {
        structure_start_contains(pos, start)
    })
}

#[cfg(test)]
mod tests {
    use steel_registry::init_vanilla_registry;
    use steel_registry::structure::TerrainAdjustment;
    use steel_utils::BoundingBox;
    use steel_worldgen::structure::StructurePiece;

    use super::{
        BlockPos, ChunkPos, ChunkStatus, Identifier, StructureStart, World, all_structures_at,
        any_start_matching, has_structure_at, references_for_structure, structure_has_piece_at,
        structure_start_contains,
    };
    use crate::chunk::chunk_holder::ChunkHolder;
    use crate::test_support::{fresh_test_world, insert_ready_full_chunk};

    /// The structure these tests install starts under, chosen because M1 already needs its key.
    fn fortress() -> Identifier {
        Identifier::vanilla_static("fortress")
    }

    /// The keys [`all_structures_at`] reports, which is the half of its answer these tests compare.
    ///
    /// Projecting to keys keeps the assertions printable: `StructureData` is `Debug` but not
    /// `PartialEq`, so the whole tuple cannot be compared against an expected list.
    fn structure_keys_at(world: &World, pos: BlockPos) -> Vec<Identifier> {
        all_structures_at(world, pos)
            .into_iter()
            .map(|(structure, _)| structure.key.clone())
            .collect()
    }

    /// A piece box spanning `min..=max`, with the metadata `non_jigsaw` gives a hand-built piece.
    ///
    /// The piece type is never read here — only placement dispatches on it — so it is named for what
    /// it is rather than borrowed from a real fortress piece.
    fn piece(min: BlockPos, max: BlockPos) -> StructurePiece {
        StructurePiece::non_jigsaw(
            Identifier::new_static("steel", "test_piece"),
            BoundingBox::from_corners(min, max),
            0,
            None,
        )
    }

    /// Vanilla `setStartForStructure`: writes a start into the chunk it originates in.
    ///
    /// [`TerrainAdjustment::None`] keeps `bb_inflate` at zero, so the start's box is exactly the
    /// union of the pieces handed in and the assertions below can name its corners.
    fn write_start(holder: &ChunkHolder, structure: &Identifier, pieces: Vec<StructurePiece>) {
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
    }

    /// Vanilla `addReferenceForStructure`: points one chunk at a start's origin chunk.
    ///
    /// The reads only ever find a start by following a reference, so a test that wrote the start
    /// alone would find nothing no matter where the boxes lay.
    fn write_reference(holder: &ChunkHolder, structure: &Identifier, origin: ChunkPos) {
        let chunk = holder
            .try_chunk(ChunkStatus::Full)
            .expect("the test chunk should be published at Full");

        chunk
            .structure_references_mut()
            .entry(structure.clone())
            .or_default()
            .insert(origin);
    }

    /// A chunk nothing generated in reports no structures at all.
    #[test]
    fn a_chunk_with_no_references_holds_no_structures() {
        init_vanilla_registry();
        let world = fresh_test_world("structure_manager_empty");
        insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
        let pos = BlockPos::new(8, 64, 8);

        assert_eq!(structure_keys_at(&world, pos), []);
        assert!(!has_structure_at(&world, pos, &fortress()));
    }

    /// A position in an unloaded chunk answers empty rather than generating one.
    ///
    /// This is Steel's one divergence from vanilla, which would generate the chunk to
    /// `STRUCTURE_REFERENCES` to answer. Pinning it keeps the difference deliberate.
    #[test]
    fn an_unloaded_chunk_answers_without_generating() {
        init_vanilla_registry();
        let world = fresh_test_world("structure_manager_unloaded");
        let pos = BlockPos::new(8, 64, 8);

        assert_eq!(structure_keys_at(&world, pos), []);
        assert_eq!(references_for_structure(&world, pos, &fortress()), []);
        assert!(!has_structure_at(&world, pos, &fortress()));
    }

    /// A reference whose origin chunk holds no start for that structure finds nothing.
    ///
    /// Vanilla's `fillStartsForStructure` tests `start != null` before anything else, because a
    /// reference outlives the start it points at whenever the origin chunk is reloaded empty.
    #[test]
    fn a_reference_without_a_start_finds_nothing() {
        init_vanilla_registry();
        let world = fresh_test_world("structure_manager_dangling");
        let chunk = insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
        write_reference(&chunk, &fortress(), ChunkPos::new(0, 0));

        assert!(!has_structure_at(
            &world,
            BlockPos::new(8, 64, 8),
            &fortress()
        ));
    }

    /// A start with no pieces is vanilla's invalid start, and no read accepts it.
    #[test]
    fn a_start_with_no_pieces_is_invalid() {
        init_vanilla_registry();
        let world = fresh_test_world("structure_manager_invalid");
        let chunk = insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
        write_start(&chunk, &fortress(), Vec::new());
        write_reference(&chunk, &fortress(), ChunkPos::new(0, 0));

        assert!(!has_structure_at(
            &world,
            BlockPos::new(8, 64, 8),
            &fortress()
        ));
    }

    /// The `full` box is the piece union, so it covers the gap between two pieces and `piece` does
    /// not.
    ///
    /// This is the whole reason a structure's `spawn_overrides` entry declares a mode: the two
    /// answers differ for every position a structure encloses but does not occupy.
    #[test]
    fn the_full_box_covers_gaps_the_piece_boxes_do_not() {
        init_vanilla_registry();
        let world = fresh_test_world("structure_manager_gap");
        let chunk = insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
        let pieces = vec![
            piece(BlockPos::new(0, 64, 0), BlockPos::new(3, 67, 3)),
            piece(BlockPos::new(12, 64, 12), BlockPos::new(15, 67, 15)),
        ];
        write_start(&chunk, &fortress(), pieces);
        write_reference(&chunk, &fortress(), ChunkPos::new(0, 0));

        let origins = references_for_structure(&world, BlockPos::new(8, 64, 8), &fortress());
        let gap = BlockPos::new(8, 64, 8);

        assert!(any_start_matching(&world, &fortress(), &origins, |start| {
            structure_start_contains(gap, start)
        }));
        assert!(!any_start_matching(
            &world,
            &fortress(),
            &origins,
            |start| { structure_has_piece_at(gap, start) }
        ));
        // The gap is inside the union, so the `full`-mode read accepts it.
        assert!(has_structure_at(&world, gap, &fortress()));
    }

    /// Both box tests are inclusive at each end, matching vanilla `BoundingBox.isInside`.
    #[test]
    fn the_boxes_include_their_own_corners() {
        init_vanilla_registry();
        let world = fresh_test_world("structure_manager_corners");
        let chunk = insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
        let min = BlockPos::new(2, 64, 2);
        let max = BlockPos::new(5, 67, 5);
        write_start(&chunk, &fortress(), vec![piece(min, max)]);
        write_reference(&chunk, &fortress(), ChunkPos::new(0, 0));

        assert!(has_structure_at(&world, min, &fortress()));
        assert!(has_structure_at(&world, max, &fortress()));
        assert!(!has_structure_at(&world, max.above(), &fortress()));
        assert!(!has_structure_at(&world, min.below(), &fortress()));
    }

    /// A start reaches the neighboring chunks that reference it, not just its own.
    ///
    /// Vanilla stores the start once, in the chunk it originated in, and every chunk the structure
    /// touches carries a reference back to that origin. A read that only looked in the queried
    /// chunk's own starts would miss most of every structure.
    #[test]
    fn a_start_is_found_through_a_neighbors_reference() {
        init_vanilla_registry();
        let world = fresh_test_world("structure_manager_neighbour");
        let origin_pos = ChunkPos::new(-1, 0);
        let origin = insert_ready_full_chunk(&world, origin_pos);
        let queried = insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
        // Straddles the chunk border: x = -4..=3 covers the last four columns of the origin chunk
        // and the first four of the queried one.
        write_start(
            &origin,
            &fortress(),
            vec![piece(BlockPos::new(-4, 64, 0), BlockPos::new(3, 67, 7))],
        );
        write_reference(&queried, &fortress(), origin_pos);

        assert!(has_structure_at(
            &world,
            BlockPos::new(2, 65, 2),
            &fortress()
        ));
        // Still inside the box, but that column belongs to the origin chunk, which never referenced
        // itself — so this read finds nothing, exactly as vanilla's would.
        assert!(!has_structure_at(
            &world,
            BlockPos::new(-2, 65, 2),
            &fortress()
        ));
    }

    /// `all_structures_at` resolves each id through the registry and drops what it cannot.
    ///
    /// The unresolvable key is the interesting half: a chunk saved under a datapack that has since
    /// been removed still names structures the registry no longer holds, and the override scan has
    /// no settings to consult for one of those.
    #[test]
    fn unresolvable_structure_ids_are_dropped() {
        init_vanilla_registry();
        let world = fresh_test_world("structure_manager_unknown");
        let chunk = insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
        let origin = ChunkPos::new(0, 0);
        write_reference(&chunk, &fortress(), origin);
        write_reference(
            &chunk,
            &Identifier::new_static("steel", "not_a_structure"),
            origin,
        );

        let found = all_structures_at(&world, BlockPos::new(8, 64, 8));

        assert_eq!(found.len(), 1);
        let (structure, origins) = &found[0];
        assert_eq!(structure.key, fortress());
        assert_eq!(origins.as_slice(), [origin]);
    }

    /// `references_for_structure` answers for one key, ignoring every other structure's origins.
    #[test]
    fn references_are_read_per_structure() {
        init_vanilla_registry();
        let world = fresh_test_world("structure_manager_per_key");
        let chunk = insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
        let monument = Identifier::vanilla_static("monument");
        write_reference(&chunk, &fortress(), ChunkPos::new(0, 0));
        write_reference(&chunk, &monument, ChunkPos::new(1, 1));

        let pos = BlockPos::new(8, 64, 8);

        assert_eq!(
            references_for_structure(&world, pos, &fortress()).as_slice(),
            [ChunkPos::new(0, 0)]
        );
        assert_eq!(
            references_for_structure(&world, pos, &monument).as_slice(),
            [ChunkPos::new(1, 1)]
        );
        assert_eq!(
            references_for_structure(&world, pos, &Identifier::vanilla_static("igloo")),
            []
        );
    }
}
