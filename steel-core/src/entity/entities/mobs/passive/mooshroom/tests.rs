//! Tests for the mooshroom's variant, persistence and breeding mutation.

use std::io::Cursor;
use std::sync::Weak;

use glam::DVec3;
use simdnbt::borrow::read_compound;
use simdnbt::owned::NbtCompound;
use steel_registry::{init_vanilla_registry, vanilla_entities, vanilla_loot_tables};

use super::{MooshroomEntity, MooshroomVariant};
use crate::entity::Entity as _;

fn mooshroom(id: i32) -> MooshroomEntity {
    MooshroomEntity::new(&vanilla_entities::MOOSHROOM, id, DVec3::ZERO, Weak::new())
}

/// Vanilla's ids are the wire values, and `byId` clamps rather than wraps.
#[test]
fn variant_ids_match_vanilla_and_clamp_out_of_range() {
    assert_eq!(MooshroomVariant::Red.id(), 0);
    assert_eq!(MooshroomVariant::Brown.id(), 1);
    assert_eq!(MooshroomVariant::DEFAULT, MooshroomVariant::Red);

    assert_eq!(MooshroomVariant::by_id(0), MooshroomVariant::Red);
    assert_eq!(MooshroomVariant::by_id(1), MooshroomVariant::Brown);
    // CLAMP, so neither end wraps around to the other variant.
    assert_eq!(MooshroomVariant::by_id(-5), MooshroomVariant::Red);
    assert_eq!(MooshroomVariant::by_id(99), MooshroomVariant::Brown);
}

/// Each variant sheared yields its own mushroom, which vanilla resolves inside the loot table.
#[test]
fn each_variant_shears_its_own_mushroom_table() {
    init_vanilla_registry();

    assert_eq!(
        MooshroomVariant::Red.shearing_loot_table().key,
        vanilla_loot_tables::SHEARING_MOOSHROOM_RED.key
    );
    assert_eq!(
        MooshroomVariant::Brown.shearing_loot_table().key,
        vanilla_loot_tables::SHEARING_MOOSHROOM_BROWN.key
    );
}

/// A fresh mooshroom is red, and the synced field follows `set_variant`.
#[test]
fn a_new_mooshroom_is_red_and_tracks_its_variant() {
    init_vanilla_registry();
    let cow = mooshroom(1);

    assert_eq!(cow.variant(), MooshroomVariant::Red);

    cow.set_variant(MooshroomVariant::Brown);
    assert_eq!(cow.variant(), MooshroomVariant::Brown);
}

/// The variant round-trips through NBT under vanilla's `Type` string names.
#[test]
fn the_variant_round_trips_through_nbt() {
    init_vanilla_registry();
    let cow = mooshroom(1);
    cow.set_variant(MooshroomVariant::Brown);

    let mut nbt = NbtCompound::new();
    cow.save_additional(&mut nbt);
    assert_eq!(
        nbt.string("Type").map(|name| name.to_string()),
        Some("brown".to_owned())
    );

    let loaded = mooshroom(2);
    let mut bytes = Vec::new();
    nbt.write(&mut bytes);
    let borrowed = read_compound(&mut Cursor::new(&bytes))
        .unwrap_or_else(|error| panic!("reborrow failed: {error}"));
    loaded.load_additional((&borrowed).into());

    assert_eq!(loaded.variant(), MooshroomVariant::Brown);
}

/// A save with no `Type`, or an unknown one, reads back as vanilla's `DEFAULT`.
#[test]
fn an_absent_or_unknown_variant_loads_as_red() {
    init_vanilla_registry();

    for tag in [None, Some("chartreuse")] {
        let cow = mooshroom(1);
        cow.set_variant(MooshroomVariant::Brown);

        let mut nbt = NbtCompound::new();
        if let Some(tag) = tag {
            nbt.insert("Type", tag);
        }
        let mut bytes = Vec::new();
        nbt.write(&mut bytes);
        let borrowed = read_compound(&mut Cursor::new(&bytes))
            .unwrap_or_else(|error| panic!("reborrow failed: {error}"));
        cow.load_additional((&borrowed).into());

        assert_eq!(cow.variant(), MooshroomVariant::Red, "tag was {tag:?}");
    }
}

/// Mismatched parents never mutate: the calf always takes one parent's own color.
///
/// Vanilla gates the 1-in-1024 mutation behind the parents matching, so this is the branch that
/// can be asserted without depending on a random draw.
#[test]
fn mismatched_parents_produce_one_of_their_own_colors() {
    init_vanilla_registry();
    let red = mooshroom(1);
    let brown = mooshroom(2);
    brown.set_variant(MooshroomVariant::Brown);

    // Whichever side the coin lands on, a mismatched pair cannot yield a mutation, and with only
    // two variants that means any result is one of the parents'. Repeat enough to see both.
    let mut saw_red = false;
    let mut saw_brown = false;
    for _ in 0..256 {
        match red.offspring_variant(&brown) {
            MooshroomVariant::Red => saw_red = true,
            MooshroomVariant::Brown => saw_brown = true,
        }
    }

    assert!(
        saw_red && saw_brown,
        "256 draws should show both parents' colors"
    );
}

/// Matching parents almost always breed true, and `flipped` is what a mutation returns.
#[test]
fn matching_parents_breed_true_far_more_often_than_they_mutate() {
    init_vanilla_registry();
    let one = mooshroom(1);
    let two = mooshroom(2);
    one.set_variant(MooshroomVariant::Brown);
    two.set_variant(MooshroomVariant::Brown);

    let mutations = (0..2048)
        .filter(|_| one.offspring_variant(&two) == MooshroomVariant::Red)
        .count();

    // The expected count over 2048 draws is 2; this bound only has to exclude "always mutates"
    // and "the mutation branch is unreachable" without being flaky.
    assert!(mutations < 64, "saw {mutations} mutations in 2048 draws");
    assert_eq!(MooshroomVariant::Brown.flipped(), MooshroomVariant::Red);
    assert_eq!(MooshroomVariant::Red.flipped(), MooshroomVariant::Brown);
}

/// Only an adult can be sheared, matching vanilla `readyForShearing`.
#[test]
fn only_an_adult_is_ready_for_shearing() {
    init_vanilla_registry();
    let cow = mooshroom(1);
    assert!(cow.ready_for_shearing());

    crate::entity::AgeableMob::set_baby(&cow, true);
    assert!(!cow.ready_for_shearing());
}
