//! Vanilla's `OcelotAttackGoal`, the stalk-and-pounce melee the felines use.
//!
//! Ports `net.minecraft.world.entity.ai.goal.OcelotAttackGoal`. This is not
//! [`MeleeAttackGoal`](super::MeleeAttackGoal) with different numbers: it re-pathfinds every tick
//! with no recalculation cooldown, and it picks one of three speeds from the distance to the
//! target, which is what makes an ocelot creep up on a chicken and then sprint the last few
//! blocks. The three speeds are the same ones `Ocelot.customServerAiStep` reads back to choose
//! between its crouching, standing and sprinting poses.

use glam::DVec3;

use super::selector::{Goal, GoalControls};
use crate::entity::{PathfinderMob, SharedEntity};

/// Ticks between swings, vanilla's `attackTime = 20`.
const ATTACK_INTERVAL_TICKS: i32 = 20;

/// Squared distance past which the goal gives up, vanilla's bare `225.0` — fifteen blocks.
const MAX_PURSUIT_DISTANCE_SQR: f64 = 225.0;

/// Squared distance inside which the mob sprints, vanilla's bare `16.0`.
const SPRINT_DISTANCE_SQR: f64 = 16.0;

/// Speed used while stalking, matching `Ocelot::CROUCH_SPEED_MOD`.
const CROUCH_SPEED_MODIFIER: f64 = 0.6;

/// Speed used at conversational range, matching `Ocelot::WALK_SPEED_MOD`.
const WALK_SPEED_MODIFIER: f64 = 0.8;

/// Speed used to close the last few blocks, matching `Ocelot::SPRINT_SPEED_MOD`.
const SPRINT_SPEED_MODIFIER: f64 = 1.33;

/// Look speed in degrees per tick, vanilla's `setLookAt(target, 30.0F, 30.0F)`.
const LOOK_SPEED: f32 = 30.0;

/// Vanilla `OcelotAttackGoal`.
pub struct OcelotAttackGoal {
    /// The target this goal latched onto, held across ticks exactly as vanilla holds it.
    target: Option<SharedEntity>,
    /// Ticks remaining before the next swing lands.
    attack_time: i32,
}

impl OcelotAttackGoal {
    /// Creates the goal.
    #[must_use]
    pub(crate) const fn new() -> Self {
        Self {
            target: None,
            attack_time: 0,
        }
    }

    /// Vanilla's melee radius: the square of twice the mob's width.
    ///
    /// Vanilla writes `getBbWidth() * 2.0F * (getBbWidth() * 2.0F)`, which is a squared distance
    /// even though the doubling is applied before the multiplication rather than after.
    fn melee_radius_sqr(mob: &dyn PathfinderMob) -> f64 {
        let width = f64::from(mob.base().dimensions().width);
        let reach = width * 2.0;
        reach * reach
    }
}

impl Goal for OcelotAttackGoal {
    fn controls(&self) -> GoalControls {
        GoalControls::MOVE | GoalControls::LOOK
    }

    fn can_use(&mut self, mob: &dyn PathfinderMob) -> bool {
        let Some(target) = mob.target() else {
            return false;
        };

        self.target = Some(target);
        true
    }

    fn can_continue_to_use(&mut self, mob: &dyn PathfinderMob) -> bool {
        let Some(target) = self.target.clone() else {
            return false;
        };

        if !target.is_alive() {
            return false;
        }

        if mob.position().distance_squared(target.position()) > MAX_PURSUIT_DISTANCE_SQR {
            return false;
        }

        !mob.mob_base().navigation().lock().is_done() || self.can_use(mob)
    }

    fn stop(&mut self, mob: &dyn PathfinderMob) {
        self.target = None;
        mob.mob_base().navigation().lock().stop();
    }

    fn requires_update_every_tick(&self) -> bool {
        true
    }

    fn tick(&mut self, mob: &dyn PathfinderMob) {
        let Some(target) = self.target.clone() else {
            return;
        };

        let target_position = target.position();
        mob.mob_base().controls().lock().look_control.set_look_at(
            DVec3::new(target_position.x, target.get_eye_y(), target_position.z),
            LOOK_SPEED,
            LOOK_SPEED,
        );

        let melee_radius_sqr = Self::melee_radius_sqr(mob);
        let distance_sqr = mob.position().distance_squared(target_position);

        // Vanilla's ladder reads oddly because the sprint window is checked first: a target
        // between the melee radius and four blocks is sprinted at, anything further inside
        // fifteen blocks is stalked, and the walk speed is only what is left over.
        let speed_modifier =
            if distance_sqr > melee_radius_sqr && distance_sqr < SPRINT_DISTANCE_SQR {
                SPRINT_SPEED_MODIFIER
            } else if distance_sqr < MAX_PURSUIT_DISTANCE_SQR {
                CROUCH_SPEED_MODIFIER
            } else {
                WALK_SPEED_MODIFIER
            };

        mob.move_to_pos(target_position, speed_modifier);

        self.attack_time = (self.attack_time - 1).max(0);
        if distance_sqr > melee_radius_sqr {
            return;
        }

        if self.attack_time <= 0 {
            self.attack_time = ATTACK_INTERVAL_TICKS;
            if let Some(world) = mob.level() {
                let _ = mob.do_hurt_target(&world, &target);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Weak};

    use glam::DVec3;
    use steel_registry::{init_vanilla_registry, vanilla_entities};

    use super::*;
    use crate::entity::{Entity as _, Mob, entities::PigEntity};

    fn pig(id: i32, position: DVec3) -> PigEntity {
        PigEntity::new(&vanilla_entities::PIG, id, position, Weak::new())
    }

    fn shared_pig(id: i32, position: DVec3) -> SharedEntity {
        Arc::new(pig(id, position))
    }

    #[test]
    fn ocelot_attack_goal_takes_move_and_look_and_runs_every_tick() {
        let goal = OcelotAttackGoal::new();

        assert_eq!(goal.controls(), GoalControls::MOVE | GoalControls::LOOK);
        assert!(goal.requires_update_every_tick());
    }

    #[test]
    fn ocelot_attack_goal_needs_a_target_to_start() {
        init_vanilla_registry();
        let mut goal = OcelotAttackGoal::new();
        let mob = pig(1, DVec3::ZERO);

        assert!(!goal.can_use(&mob));
        assert!(goal.target.is_none());
    }

    #[test]
    fn ocelot_attack_goal_latches_onto_the_mobs_target() {
        init_vanilla_registry();
        let mut goal = OcelotAttackGoal::new();
        let mob = pig(1, DVec3::ZERO);
        let target = shared_pig(2, DVec3::new(3.0, 0.0, 0.0));
        assert!(mob.set_target(Some(&target)));

        assert!(goal.can_use(&mob));
        assert_eq!(goal.target.as_ref().map(|held| held.id()), Some(2));
    }

    #[test]
    fn ocelot_attack_goal_drops_a_target_past_fifteen_blocks() {
        init_vanilla_registry();
        let mut goal = OcelotAttackGoal::new();
        let mob = pig(1, DVec3::ZERO);
        let target = shared_pig(2, DVec3::new(16.0, 0.0, 0.0));
        goal.target = Some(target);

        assert!(!goal.can_continue_to_use(&mob));
    }

    #[test]
    fn ocelot_attack_goal_melee_radius_is_twice_the_width_squared() {
        init_vanilla_registry();
        let mob = pig(1, DVec3::ZERO);
        let width = f64::from(mob.base().dimensions().width);

        let expected = (width * 2.0) * (width * 2.0);
        assert!((OcelotAttackGoal::melee_radius_sqr(&mob) - expected).abs() <= f64::EPSILON);
    }

    #[test]
    fn ocelot_attack_goal_without_a_target_ticks_harmlessly() {
        init_vanilla_registry();
        let mut goal = OcelotAttackGoal::new();
        let mob = pig(1, DVec3::ZERO);

        goal.tick(&mob);

        assert_eq!(goal.attack_time, 0);
    }
}
