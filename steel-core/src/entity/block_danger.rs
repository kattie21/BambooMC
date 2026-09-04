//! Vanilla `EntityType.isBlockDangerous`, the per-type hazardous-block predicate.
//!
//! Vanilla asks two questions of a block state: does this entity type's `immuneTo` block tag cover
//! it, and if not, is the block one of the five hazards — a burning block the type is not fire
//! immune to, a wither rose, a sweet berry bush, a cactus or powder snow. Both natural spawning and
//! dismounting gate on the answer, so it lives here rather than in either caller.
//!
//! `immuneTo` is a per-type field on `EntityType`, set by `EntityType.Builder.immuneTo` in six of
//! `EntityTypes`' registrations and left at `#minecraft:default_immune_to` for the other 152. The
//! extracted `entities.json` does not carry it, and Steel's `EntityType` struct lives in
//! `steel-registry` while this predicate needs `WalkPathEvaluator::is_burning_block` from
//! `steel-core`, so the assignment is transcribed as a table here. This follows
//! [`steel_registry::blocks::spawn_rule::BlockSpawnRule`], which does the same for a vanilla table
//! that generated data likewise does not carry.

use std::ptr;

use steel_registry::blocks::block_state_ext::BlockStateExt as _;
use steel_registry::entity_type::EntityTypeRef;
use steel_registry::vanilla_block_tags::BlockTag;
use steel_registry::{vanilla_blocks, vanilla_entities};
use steel_utils::{BlockStateId, Identifier};

use crate::entity::ai::walk::WalkPathEvaluator;

/// Returns vanilla `EntityType.isBlockDangerous`.
///
/// The immunity tag short-circuits every hazard, and fire immunity short-circuits only the burning
/// term — vanilla's Java precedence groups the four block comparisons outside the fire-immunity
/// guard, so a fire-immune type still avoids a cactus.
#[must_use]
pub fn is_block_dangerous(entity_type: EntityTypeRef, state: BlockStateId) -> bool {
    let block = state.get_block();

    if block.has_tag(&immune_to_tag(entity_type)) {
        return false;
    }

    !entity_type.fire_immune && WalkPathEvaluator::is_burning_block(state)
        || block == &vanilla_blocks::WITHER_ROSE
        || block == &vanilla_blocks::SWEET_BERRY_BUSH
        || block == &vanilla_blocks::CACTUS
        || block == &vanilla_blocks::POWDER_SNOW
}

/// Returns the block tag this entity type is immune to.
///
/// Six of `EntityTypes`' 158 registrations call `Builder.immuneTo`: the fox at `EntityTypes:475`,
/// the polar bear at `:805`, the snow golem at `:911`, the stray at `:963`, the wither at `:1079`
/// and the wither skeleton at `:1088`. Every other type keeps the builder's default of
/// `#minecraft:default_immune_to`, which holds no blocks, so the immunity term is inert for them.
///
/// Returned by value because [`BlockTag`]'s entries are `const` items of a `Cow`-bearing
/// [`Identifier`], which cannot be promoted to a `'static` reference.
#[must_use]
pub fn immune_to_tag(entity_type: EntityTypeRef) -> Identifier {
    if ptr::eq(entity_type, &raw const vanilla_entities::FOX) {
        BlockTag::FOX_IMMUNE_TO
    } else if ptr::eq(entity_type, &raw const vanilla_entities::POLAR_BEAR) {
        BlockTag::POLAR_BEAR_IMMUNE_TO
    } else if ptr::eq(entity_type, &raw const vanilla_entities::SNOW_GOLEM) {
        BlockTag::SNOW_GOLEM_IMMUNE_TO
    } else if ptr::eq(entity_type, &raw const vanilla_entities::STRAY) {
        BlockTag::STRAY_IMMUNE_TO
    } else if ptr::eq(entity_type, &raw const vanilla_entities::WITHER) {
        BlockTag::WITHER_IMMUNE_TO
    } else if ptr::eq(entity_type, &raw const vanilla_entities::WITHER_SKELETON) {
        BlockTag::WITHER_SKELETON_IMMUNE_TO
    } else {
        BlockTag::DEFAULT_IMMUNE_TO
    }
}

#[cfg(test)]
mod tests {
    use steel_registry::init_vanilla_registry;

    use super::*;

    /// Vanilla's five hazards, in the order `EntityType.isBlockDangerous` tests them.
    ///
    /// The first stands for the burning term, which [`WalkPathEvaluator::is_burning_block`] widens
    /// to the `#minecraft:fire` tag, lava, the magma block, a lava cauldron and a lit campfire.
    fn hazards() -> [BlockStateId; 5] {
        [
            vanilla_blocks::FIRE.default_state(),
            vanilla_blocks::WITHER_ROSE.default_state(),
            vanilla_blocks::SWEET_BERRY_BUSH.default_state(),
            vanilla_blocks::CACTUS.default_state(),
            vanilla_blocks::POWDER_SNOW.default_state(),
        ]
    }

    /// An ordinary type keeps the builder default, and that tag holds no blocks.
    ///
    /// This is what makes the immunity term inert for 152 of vanilla's 158 registrations, so the
    /// four hazards below are reached rather than short-circuited.
    #[test]
    fn the_default_immunity_tag_covers_nothing() {
        init_vanilla_registry();

        assert_eq!(
            immune_to_tag(&vanilla_entities::ZOMBIE),
            BlockTag::DEFAULT_IMMUNE_TO
        );
        for state in hazards() {
            assert!(!state.get_block().has_tag(&BlockTag::DEFAULT_IMMUNE_TO));
        }
    }

    /// Every hazard endangers a type with neither an immunity tag nor fire immunity.
    #[test]
    fn all_five_hazards_are_dangerous_to_an_ordinary_type() {
        init_vanilla_registry();

        for state in hazards() {
            assert!(
                is_block_dangerous(&vanilla_entities::ZOMBIE, state),
                "{} must endanger a zombie",
                state.get_block().key
            );
        }
    }

    /// The burning term is the whole of what `is_burning_block` covers, not just the fire block.
    #[test]
    fn the_burning_term_covers_every_burning_block() {
        init_vanilla_registry();

        for state in [
            vanilla_blocks::FIRE.default_state(),
            vanilla_blocks::SOUL_FIRE.default_state(),
            vanilla_blocks::LAVA.default_state(),
            vanilla_blocks::MAGMA_BLOCK.default_state(),
            vanilla_blocks::LAVA_CAULDRON.default_state(),
        ] {
            assert!(
                WalkPathEvaluator::is_burning_block(state),
                "{} must be a burning block, or this test proves nothing",
                state.get_block().key
            );
            assert!(is_block_dangerous(&vanilla_entities::ZOMBIE, state));
        }
    }

    /// Ordinary blocks endanger nobody.
    #[test]
    fn harmless_blocks_are_never_dangerous() {
        init_vanilla_registry();

        for state in [
            vanilla_blocks::AIR.default_state(),
            vanilla_blocks::STONE.default_state(),
            vanilla_blocks::GRASS_BLOCK.default_state(),
            vanilla_blocks::WATER.default_state(),
        ] {
            for entity_type in [&vanilla_entities::ZOMBIE, &vanilla_entities::FOX] {
                assert!(!is_block_dangerous(entity_type, state));
            }
        }
    }

    /// Fire immunity spares the burning term alone.
    ///
    /// Vanilla's Java precedence groups the four block comparisons outside the `!fireImmune` guard,
    /// so reading the expression as `!fireImmune && (burning || rose || … )` would wrongly make a
    /// blaze safe on a cactus. That misreading is exactly what this pins.
    #[test]
    fn fire_immunity_spares_only_the_burning_term() {
        init_vanilla_registry();

        assert!(
            vanilla_entities::BLAZE.fire_immune,
            "the blaze must be fire immune, or this test proves nothing"
        );
        assert!(!is_block_dangerous(
            &vanilla_entities::BLAZE,
            vanilla_blocks::LAVA.default_state()
        ));
        for state in [
            vanilla_blocks::WITHER_ROSE.default_state(),
            vanilla_blocks::SWEET_BERRY_BUSH.default_state(),
            vanilla_blocks::CACTUS.default_state(),
            vanilla_blocks::POWDER_SNOW.default_state(),
        ] {
            assert!(
                is_block_dangerous(&vanilla_entities::BLAZE, state),
                "{} must still endanger a fire-immune type",
                state.get_block().key
            );
        }
    }

    /// Each of the six tabled types is immune to exactly the one block its tag holds.
    ///
    /// Every one of these tags carries a single entry, so the pairing doubles as a check that the
    /// six rows are not transposed: swapping any two would leave a type exposed to its own hazard.
    #[test]
    fn tabled_types_are_immune_to_their_own_hazard_only() {
        init_vanilla_registry();

        let rows: [(EntityTypeRef, BlockStateId); 6] = [
            (
                &vanilla_entities::FOX,
                vanilla_blocks::SWEET_BERRY_BUSH.default_state(),
            ),
            (
                &vanilla_entities::POLAR_BEAR,
                vanilla_blocks::POWDER_SNOW.default_state(),
            ),
            (
                &vanilla_entities::SNOW_GOLEM,
                vanilla_blocks::POWDER_SNOW.default_state(),
            ),
            (
                &vanilla_entities::STRAY,
                vanilla_blocks::POWDER_SNOW.default_state(),
            ),
            (
                &vanilla_entities::WITHER,
                vanilla_blocks::WITHER_ROSE.default_state(),
            ),
            (
                &vanilla_entities::WITHER_SKELETON,
                vanilla_blocks::WITHER_ROSE.default_state(),
            ),
        ];

        for (entity_type, immune_state) in rows {
            assert_ne!(
                immune_to_tag(entity_type),
                BlockTag::DEFAULT_IMMUNE_TO,
                "{} must carry its own immunity tag",
                entity_type.key
            );
            assert!(
                !is_block_dangerous(entity_type, immune_state),
                "{} must be immune to {}",
                entity_type.key,
                immune_state.get_block().key
            );
            assert!(
                is_block_dangerous(entity_type, vanilla_blocks::CACTUS.default_state()),
                "{} must not be immune to a cactus",
                entity_type.key
            );
            assert!(
                is_block_dangerous(&vanilla_entities::ZOMBIE, immune_state),
                "{} must endanger an untabled type, or the immunity proves nothing",
                immune_state.get_block().key
            );
        }
    }
}
