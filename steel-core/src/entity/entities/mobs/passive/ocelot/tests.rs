//! Vanilla-parity tests for [`OcelotEntity`](super::OcelotEntity).

use std::sync::Weak;

use glam::DVec3;
use simdnbt::owned::NbtCompound;
use steel_registry::{init_vanilla_registry, vanilla_entities};

use super::*;
use crate::entity::spawn_placements;
use steel_utils::random::legacy_random::LegacyRandom;

fn ocelot(id: i32) -> OcelotEntity {
    OcelotEntity::new(&vanilla_entities::OCELOT, id, DVec3::ZERO, Weak::new())
}

#[test]
fn a_new_ocelot_does_not_trust_anyone() {
    init_vanilla_registry();
    let ocelot = ocelot(1);

    assert!(!ocelot.is_trusting());
}

#[test]
fn the_trust_flag_round_trips_through_nbt() {
    init_vanilla_registry();
    let ocelot = ocelot(1);
    ocelot.set_trusting(true);

    let mut saved = NbtCompound::new();
    Entity::save_additional(&ocelot, &mut saved);

    assert_eq!(saved.byte("Trusting"), Some(1));
}

#[test]
fn an_absent_trust_tag_loads_as_untrusting() {
    init_vanilla_registry();
    let ocelot = ocelot(1);
    ocelot.set_trusting(true);

    let bytes = {
        let mut bytes = Vec::new();
        NbtCompound::new().write(&mut bytes);
        bytes
    };
    let borrowed = simdnbt::borrow::read_compound(&mut std::io::Cursor::new(&bytes))
        .unwrap_or_else(|error| panic!("test nbt should reborrow: {error}"));
    Entity::load_additional(&ocelot, (&borrowed).into());

    assert!(!ocelot.is_trusting());
}

#[test]
fn a_wild_ocelot_despawns_only_after_the_grace_period() {
    init_vanilla_registry();
    let ocelot = ocelot(1);

    assert!(!Mob::remove_when_far_away(&ocelot, 1.0e6));
}

#[test]
fn a_trusting_ocelot_never_despawns() {
    init_vanilla_registry();
    let ocelot = ocelot(1);
    ocelot.set_trusting(true);

    assert!(!Mob::remove_when_far_away(&ocelot, 1.0e6));
}

#[test]
fn a_baby_ocelot_is_smaller_than_an_adult() {
    init_vanilla_registry();
    let ocelot = ocelot(1);
    let adult = Entity::dimensions_for_pose(&ocelot, EntityPose::Standing);

    AgeableMob::set_baby(&ocelot, true);
    let baby = Entity::dimensions_for_pose(&ocelot, EntityPose::Standing);

    assert!(baby.width < adult.width);
    assert!(baby.height < adult.height);
    assert_eq!(baby.width.to_bits(), OCELOT_BABY_WIDTH.to_bits());
    assert_eq!(baby.height.to_bits(), OCELOT_BABY_HEIGHT.to_bits());
}

#[test]
fn the_ambient_interval_matches_vanilla() {
    init_vanilla_registry();
    let ocelot = ocelot(1);

    assert_eq!(Mob::ambient_sound_interval(&ocelot), 900);
}

#[test]
fn the_three_gaits_keep_vanillas_exact_literals() {
    assert!((CROUCH_SPEED_MODIFIER - 0.6).abs() <= f64::EPSILON);
    assert!((WALK_SPEED_MODIFIER - 0.8).abs() <= f64::EPSILON);
    assert!((SPRINT_SPEED_MODIFIER - 1.33).abs() <= f64::EPSILON);
}

#[test]
fn the_spawn_rule_refuses_about_one_attempt_in_three() {
    init_vanilla_registry();
    let mut random = LegacyRandom::from_seed(20260903);

    let mut accepted = 0;
    for _ in 0..3000 {
        if spawn_placements::check_ocelot_spawn_rules(&mut random) {
            accepted += 1;
        }
    }

    // Two in three, with room for sampling noise on 3000 draws.
    assert!(
        (1900..=2100).contains(&accepted),
        "expected about 2000 of 3000 attempts to pass, got {accepted}"
    );
}

#[test]
fn the_placement_row_now_carries_the_ocelot_predicate() {
    init_vanilla_registry();

    assert_eq!(
        spawn_placements::heightmap_type(&vanilla_entities::OCELOT),
        crate::chunk::heightmap::HeightmapType::MotionBlocking
    );
}
