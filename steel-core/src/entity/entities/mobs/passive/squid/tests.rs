use std::f32::consts::{PI, TAU};

use steel_registry::{init_vanilla_registry, vanilla_attributes, vanilla_entities};

use super::*;

/// Trials for the randomized draws, enough to catch a bad range without slowing the suite.
const RANDOM_TRIALS: u32 = 64;

/// Vanilla's full tentacle turn, the value `advance_tentacle_cycle` wraps against.
const FULL_TURN: f32 = TAU;

fn squid() -> SquidEntity {
    init_vanilla_registry();
    SquidEntity::new(&vanilla_entities::SQUID, 1, DVec3::ZERO, Weak::new())
}

#[test]
fn squid_initializes_vanilla_living_attributes_and_health() {
    let squid = squid();

    assert_eq!(
        squid.get_health().to_bits(),
        squid.get_max_health().to_bits()
    );
    let attributes = squid.attributes().lock();
    assert_eq!(
        attributes
            .required_value(vanilla_attributes::MAX_HEALTH)
            .to_bits(),
        10.0_f64.to_bits()
    );
}

#[test]
fn squid_uses_vanilla_reduced_gravity() {
    let squid = squid();

    assert_eq!(squid.get_gravity().to_bits(), SQUID_GRAVITY.to_bits());
}

#[test]
fn squid_registers_both_vanilla_goals_at_their_priorities() {
    let squid = squid();

    // Vanilla adds the random-movement goal at priority 0 and the flee goal at priority 1.
    assert_eq!(
        squid
            .mob_base
            .goal_selector()
            .lock()
            .available_goal_priorities(),
        vec![0, 1]
    );
}

#[test]
fn a_fresh_squid_has_no_thrust_yet() {
    let squid = squid();

    // Vanilla starts `movementVector` at `Vec3.ZERO`, so the random-movement goal rolls one on its
    // first tick rather than the constructor picking a direction.
    assert!(!squid.has_movement_vector());
}

#[test]
fn water_pathfinding_is_free_for_a_squid() {
    let squid = squid();

    let malus = squid
        .mob_base
        .pathfinding_malus()
        .lock()
        .get(crate::entity::ai::path::PathType::Water);

    assert_eq!(malus.to_bits(), 0.0_f32.to_bits());
}

#[test]
fn a_baby_squid_uses_the_smaller_dimensions() {
    let squid = squid();

    let adult = squid.dimensions_for_pose(EntityPose::Standing);
    AgeableMob::set_baby(&squid, true);
    let baby = squid.dimensions_for_pose(EntityPose::Standing);

    assert_eq!(baby.width.to_bits(), SQUID_BABY_WIDTH.to_bits());
    assert_eq!(baby.height.to_bits(), SQUID_BABY_HEIGHT.to_bits());
    assert!(baby.width < adult.width);
}

#[test]
fn the_tentacle_speed_stays_inside_vanillas_draw() {
    for _ in 0..RANDOM_TRIALS {
        let speed = SquidAnimationState::roll_tentacle_speed();

        // Vanilla draws `1.0F / (nextFloat() + 1.0F) * 0.2F`, so the value falls in (0.1, 0.2].
        assert!(speed > 0.1, "tentacle speed {speed} should exceed 0.1");
        assert!(speed <= 0.2, "tentacle speed {speed} should not exceed 0.2");
    }
}

#[test]
fn the_tentacle_cycle_reports_its_wrap_and_subtracts_a_full_turn() {
    let squid = squid();

    {
        let mut animation = squid.animation.lock();
        // Park the phase just under a full turn with a known speed so the next advance must wrap.
        animation.tentacle_speed = 0.2;
        animation.tentacle_movement = FULL_TURN - 0.1;
    }

    assert!(squid.advance_tentacle_cycle());

    let animation = squid.animation.lock();
    // Vanilla subtracts one full turn on the server rather than clamping, so the remainder carries.
    let expected = FULL_TURN - 0.1 + 0.2 - FULL_TURN;
    assert!(
        (animation.tentacle_movement - expected).abs() < 1.0e-6,
        "phase should carry the remainder, got {}",
        animation.tentacle_movement
    );
    assert!(animation.tentacle_movement < PI);
}

#[test]
fn the_tentacle_cycle_stays_quiet_inside_one_turn() {
    let squid = squid();

    {
        let mut animation = squid.animation.lock();
        animation.tentacle_speed = 0.2;
        animation.tentacle_movement = 1.0;
    }

    assert!(!squid.advance_tentacle_cycle());

    let animation = squid.animation.lock();
    assert!((animation.tentacle_movement - 1.2).abs() < 1.0e-6);
    // The previous-tick snapshots move in step, which is what the ink orientation reads.
    assert!((animation.old_tentacle_movement - 1.0).abs() < 1.0e-6);
}

#[test]
fn rotating_by_zero_leaves_a_vector_alone() {
    let vector = DVec3::new(0.25, -1.0, 0.5);

    assert_eq!(rotate_x(vector, 0.0), vector);
    assert_eq!(rotate_y(vector, 0.0), vector);
}

#[test]
fn a_quarter_turn_about_x_sends_down_to_north() {
    let rotated = rotate_x(DVec3::new(0.0, -1.0, 0.0), PI / 2.0);

    assert!(
        rotated.y.abs() < 1.0e-6,
        "y should vanish, got {}",
        rotated.y
    );
    assert!(
        (rotated.z - 1.0).abs() < 1.0e-6,
        "down should rotate onto +z, got {}",
        rotated.z
    );
}

#[test]
fn a_quarter_turn_about_y_sends_east_to_south() {
    let rotated = rotate_y(DVec3::new(1.0, 0.0, 0.0), PI / 2.0);

    assert!(
        rotated.x.abs() < 1.0e-6,
        "x should vanish, got {}",
        rotated.x
    );
    assert!(
        (rotated.z + 1.0).abs() < 1.0e-6,
        "east should rotate onto -z, got {}",
        rotated.z
    );
}

#[test]
fn an_unrotated_squid_inks_straight_down() {
    let squid = squid();

    // Both body angles start at zero, so vanilla's body-local "down" stays world-down.
    let origin = squid.rotate_vector(DVec3::new(0.0, -1.0, 0.0));

    assert!(origin.x.abs() < 1.0e-6);
    assert!((origin.y + 1.0).abs() < 1.0e-6);
    assert!(origin.z.abs() < 1.0e-6);
}
