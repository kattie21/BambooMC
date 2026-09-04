//! Vanilla's per-block `isValidSpawn` predicate table.
//!
//! `BlockBehaviour.Properties` carries an `isValidSpawn` predicate whose default asks that the
//! block's upward face be sturdy and that the block emit a light level below 14. Thirty-four
//! registrations in `Blocks` replace that default, and between them they use only five distinct
//! predicates. This module reproduces the assignment; the default itself needs a level and a
//! position, so it stays with the caller.
//!
//! The predicate hangs off `Properties` rather than off a block class, which is why it cannot be a
//! block-behavior override: `bedrock` is a plain `Block` and `glass` a `TransparentBlock`, classes
//! shared with hundreds of blocks that keep the default. It is also absent from the extracted
//! `blocks.json`, so it cannot come from generated data.
//!
//! Vanilla block tags come close to three of the sets, and are deliberately not used. Adding a
//! block to `minecraft:trapdoors` does not change its spawn rule in vanilla, so keying off the tag
//! would invent a divergence that is merely latent today.

use std::ptr;

use crate::blocks::BlockRef;
use crate::entity_type::EntityTypeRef;
use crate::{vanilla_blocks, vanilla_entities};

/// Which vanilla `isValidSpawn` predicate a block carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockSpawnRule {
    /// Vanilla's default: the upward face must be sturdy and the block must emit less than light
    /// level 14. Every block that does not override the predicate carries this.
    SturdyAndDim,
    /// Refuses every entity type; vanilla's `Blocks.never`.
    Never,
    /// Accepts every entity type; vanilla's `Blocks.always`.
    Always,
    /// Accepts ocelots and parrots only; vanilla's `Blocks.ocelotOrParrot`, carried by leaves.
    OcelotOrParrot,
    /// Accepts polar bears only; carried by ice and frosted ice.
    PolarBearOnly,
    /// Accepts fire-immune types only; carried by the magma block.
    FireImmuneOnly,
}

impl BlockSpawnRule {
    /// Returns the rule this block carries.
    #[must_use]
    pub fn of(block: BlockRef) -> Self {
        if contains(NEVER_BLOCKS, block) {
            Self::Never
        } else if contains(ALWAYS_BLOCKS, block) {
            Self::Always
        } else if contains(OCELOT_OR_PARROT_BLOCKS, block) {
            Self::OcelotOrParrot
        } else if contains(POLAR_BEAR_BLOCKS, block) {
            Self::PolarBearOnly
        } else if ptr::eq(block, &raw const vanilla_blocks::MAGMA_BLOCK) {
            Self::FireImmuneOnly
        } else {
            Self::SturdyAndDim
        }
    }

    /// Applies the rule to one entity type.
    ///
    /// Returns `None` for [`Self::SturdyAndDim`], which the caller must evaluate itself because
    /// vanilla's default predicate reads the level and the position.
    #[must_use]
    pub fn allows(self, entity_type: EntityTypeRef) -> Option<bool> {
        match self {
            Self::SturdyAndDim => None,
            Self::Never => Some(false),
            Self::Always => Some(true),
            Self::OcelotOrParrot => Some(
                ptr::eq(entity_type, &raw const vanilla_entities::OCELOT)
                    || ptr::eq(entity_type, &raw const vanilla_entities::PARROT),
            ),
            Self::PolarBearOnly => Some(ptr::eq(
                entity_type,
                &raw const vanilla_entities::POLAR_BEAR,
            )),
            Self::FireImmuneOnly => Some(entity_type.fire_immune),
        }
    }
}

/// Identity search over one of the tables below.
///
/// Vanilla compares block references, so this compares addresses rather than keys. The tables are
/// short enough that a scan beats hashing, and the whole lookup runs once per spawn candidate.
fn contains(table: &[BlockRef], block: BlockRef) -> bool {
    table.iter().any(|entry| ptr::eq(*entry, block))
}

/// Blocks that refuse every spawn.
///
/// Twenty-three `Blocks` registrations, expanded through the color and weathering collections into
/// these entries: bedrock, glass, the moving piston, all sixteen stained glasses, all twelve wooden
/// trapdoors, the barrier, the iron trapdoor, the chorus flower, scaffolding, tinted glass, all
/// eight copper trapdoors and all eight copper grates.
static NEVER_BLOCKS: &[BlockRef] = &[
    &vanilla_blocks::BEDROCK,
    &vanilla_blocks::GLASS,
    &vanilla_blocks::MOVING_PISTON,
    &vanilla_blocks::WHITE_STAINED_GLASS,
    &vanilla_blocks::ORANGE_STAINED_GLASS,
    &vanilla_blocks::MAGENTA_STAINED_GLASS,
    &vanilla_blocks::LIGHT_BLUE_STAINED_GLASS,
    &vanilla_blocks::YELLOW_STAINED_GLASS,
    &vanilla_blocks::LIME_STAINED_GLASS,
    &vanilla_blocks::PINK_STAINED_GLASS,
    &vanilla_blocks::GRAY_STAINED_GLASS,
    &vanilla_blocks::LIGHT_GRAY_STAINED_GLASS,
    &vanilla_blocks::CYAN_STAINED_GLASS,
    &vanilla_blocks::PURPLE_STAINED_GLASS,
    &vanilla_blocks::BLUE_STAINED_GLASS,
    &vanilla_blocks::BROWN_STAINED_GLASS,
    &vanilla_blocks::GREEN_STAINED_GLASS,
    &vanilla_blocks::RED_STAINED_GLASS,
    &vanilla_blocks::BLACK_STAINED_GLASS,
    &vanilla_blocks::TINTED_GLASS,
    &vanilla_blocks::BARRIER,
    &vanilla_blocks::CHORUS_FLOWER,
    &vanilla_blocks::SCAFFOLDING,
    &vanilla_blocks::OAK_TRAPDOOR,
    &vanilla_blocks::SPRUCE_TRAPDOOR,
    &vanilla_blocks::BIRCH_TRAPDOOR,
    &vanilla_blocks::JUNGLE_TRAPDOOR,
    &vanilla_blocks::ACACIA_TRAPDOOR,
    &vanilla_blocks::CHERRY_TRAPDOOR,
    &vanilla_blocks::DARK_OAK_TRAPDOOR,
    &vanilla_blocks::PALE_OAK_TRAPDOOR,
    &vanilla_blocks::MANGROVE_TRAPDOOR,
    &vanilla_blocks::BAMBOO_TRAPDOOR,
    &vanilla_blocks::CRIMSON_TRAPDOOR,
    &vanilla_blocks::WARPED_TRAPDOOR,
    &vanilla_blocks::IRON_TRAPDOOR,
    &vanilla_blocks::COPPER_TRAPDOOR,
    &vanilla_blocks::EXPOSED_COPPER_TRAPDOOR,
    &vanilla_blocks::WEATHERED_COPPER_TRAPDOOR,
    &vanilla_blocks::OXIDIZED_COPPER_TRAPDOOR,
    &vanilla_blocks::WAXED_COPPER_TRAPDOOR,
    &vanilla_blocks::WAXED_EXPOSED_COPPER_TRAPDOOR,
    &vanilla_blocks::WAXED_WEATHERED_COPPER_TRAPDOOR,
    &vanilla_blocks::WAXED_OXIDIZED_COPPER_TRAPDOOR,
    &vanilla_blocks::COPPER_GRATE,
    &vanilla_blocks::EXPOSED_COPPER_GRATE,
    &vanilla_blocks::WEATHERED_COPPER_GRATE,
    &vanilla_blocks::OXIDIZED_COPPER_GRATE,
    &vanilla_blocks::WAXED_COPPER_GRATE,
    &vanilla_blocks::WAXED_EXPOSED_COPPER_GRATE,
    &vanilla_blocks::WAXED_WEATHERED_COPPER_GRATE,
    &vanilla_blocks::WAXED_OXIDIZED_COPPER_GRATE,
];

/// Blocks that accept every spawn regardless of light or face support.
static ALWAYS_BLOCKS: &[BlockRef] = &[
    &vanilla_blocks::SOUL_SAND,
    &vanilla_blocks::CARVED_PUMPKIN,
    &vanilla_blocks::JACK_O_LANTERN,
    &vanilla_blocks::REDSTONE_LAMP,
    &vanilla_blocks::MUD,
];

/// Blocks that accept ocelots and parrots only.
///
/// Nine of these take the shared leaf properties; cherry and pale oak leaves repeat them inline.
/// The set is exactly `minecraft:leaves`, which is a coincidence and not the mechanism.
static OCELOT_OR_PARROT_BLOCKS: &[BlockRef] = &[
    &vanilla_blocks::OAK_LEAVES,
    &vanilla_blocks::SPRUCE_LEAVES,
    &vanilla_blocks::BIRCH_LEAVES,
    &vanilla_blocks::JUNGLE_LEAVES,
    &vanilla_blocks::ACACIA_LEAVES,
    &vanilla_blocks::CHERRY_LEAVES,
    &vanilla_blocks::DARK_OAK_LEAVES,
    &vanilla_blocks::PALE_OAK_LEAVES,
    &vanilla_blocks::MANGROVE_LEAVES,
    &vanilla_blocks::AZALEA_LEAVES,
    &vanilla_blocks::FLOWERING_AZALEA_LEAVES,
];

/// Blocks that accept polar bears only.
static POLAR_BEAR_BLOCKS: &[BlockRef] = &[&vanilla_blocks::ICE, &vanilla_blocks::FROSTED_ICE];

#[cfg(test)]
mod tests {
    use super::{
        ALWAYS_BLOCKS, BlockSpawnRule, NEVER_BLOCKS, OCELOT_OR_PARROT_BLOCKS, POLAR_BEAR_BLOCKS,
    };
    use crate::blocks::BlockRef;
    use crate::{vanilla_blocks, vanilla_entities};
    use std::ptr;

    /// Every table size, pinned against the `Blocks` registrations they were read from.
    ///
    /// Twenty-three `never` call sites expand to 52 blocks because three of them sit on color and
    /// weathering collections; miscounting that expansion is the failure this guards.
    #[test]
    fn table_sizes_match_the_vanilla_registrations() {
        assert_eq!(NEVER_BLOCKS.len(), 52);
        assert_eq!(ALWAYS_BLOCKS.len(), 5);
        assert_eq!(OCELOT_OR_PARROT_BLOCKS.len(), 11);
        assert_eq!(POLAR_BEAR_BLOCKS.len(), 2);
    }

    /// No block may carry two rules, and no table may repeat a block.
    ///
    /// `BlockSpawnRule::of` resolves the tables in order, so a duplicate would silently shadow
    /// rather than fail.
    #[test]
    fn tables_are_disjoint_and_free_of_repeats() {
        let tables: [&[BlockRef]; 4] = [
            NEVER_BLOCKS,
            ALWAYS_BLOCKS,
            OCELOT_OR_PARROT_BLOCKS,
            POLAR_BEAR_BLOCKS,
        ];
        let mut seen: Vec<BlockRef> = Vec::new();
        for table in tables {
            for block in table {
                assert!(
                    !seen.iter().any(|other| ptr::eq(*other, *block)),
                    "{} appears in more than one spawn rule table",
                    block.key
                );
                seen.push(block);
            }
        }
        assert_eq!(seen.len(), 70);
        assert!(
            !seen
                .iter()
                .any(|block| ptr::eq(*block, &raw const vanilla_blocks::MAGMA_BLOCK)),
            "the magma block carries its own rule and must not be tabled"
        );
    }

    /// An ordinary block keeps vanilla's default, and the default refuses to answer on its own.
    #[test]
    fn ordinary_blocks_defer_to_the_default_predicate() {
        let rule = BlockSpawnRule::of(&vanilla_blocks::STONE);
        assert_eq!(rule, BlockSpawnRule::SturdyAndDim);
        assert_eq!(rule.allows(&vanilla_entities::ZOMBIE), None);
        assert_eq!(
            BlockSpawnRule::of(&vanilla_blocks::DIRT),
            BlockSpawnRule::SturdyAndDim
        );
    }

    /// The two unconditional rules answer the same way for every entity type.
    #[test]
    fn never_and_always_ignore_the_entity_type() {
        for entity_type in [
            &vanilla_entities::ZOMBIE,
            &vanilla_entities::COW,
            &vanilla_entities::OCELOT,
            &vanilla_entities::POLAR_BEAR,
            &vanilla_entities::BLAZE,
        ] {
            assert_eq!(
                BlockSpawnRule::of(&vanilla_blocks::BEDROCK).allows(entity_type),
                Some(false)
            );
            assert_eq!(
                BlockSpawnRule::of(&vanilla_blocks::SOUL_SAND).allows(entity_type),
                Some(true)
            );
        }
    }

    /// Leaves are the ocelot and parrot rule, and nothing else passes.
    #[test]
    fn leaves_accept_only_ocelots_and_parrots() {
        for block in OCELOT_OR_PARROT_BLOCKS {
            let rule = BlockSpawnRule::of(block);
            assert_eq!(rule, BlockSpawnRule::OcelotOrParrot, "{}", block.key);
            assert_eq!(rule.allows(&vanilla_entities::OCELOT), Some(true));
            assert_eq!(rule.allows(&vanilla_entities::PARROT), Some(true));
            assert_eq!(rule.allows(&vanilla_entities::COW), Some(false));
            assert_eq!(rule.allows(&vanilla_entities::ZOMBIE), Some(false));
        }
    }

    /// Ice and frosted ice are the only blocks polar bears alone may spawn on.
    #[test]
    fn ice_accepts_only_polar_bears() {
        for block in POLAR_BEAR_BLOCKS {
            let rule = BlockSpawnRule::of(block);
            assert_eq!(rule, BlockSpawnRule::PolarBearOnly, "{}", block.key);
            assert_eq!(rule.allows(&vanilla_entities::POLAR_BEAR), Some(true));
            assert_eq!(rule.allows(&vanilla_entities::COW), Some(false));
        }
    }

    /// The magma block reads the spawning type's own fire immunity rather than a fixed list.
    #[test]
    fn magma_block_accepts_fire_immune_types_only() {
        let rule = BlockSpawnRule::of(&vanilla_blocks::MAGMA_BLOCK);
        assert_eq!(rule, BlockSpawnRule::FireImmuneOnly);
        assert!(vanilla_entities::BLAZE.fire_immune);
        assert!(!vanilla_entities::COW.fire_immune);
        assert_eq!(rule.allows(&vanilla_entities::BLAZE), Some(true));
        assert_eq!(rule.allows(&vanilla_entities::MAGMA_CUBE), Some(true));
        assert_eq!(rule.allows(&vanilla_entities::COW), Some(false));
    }
}
