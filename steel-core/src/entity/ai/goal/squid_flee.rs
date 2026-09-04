//! Vanilla `Squid.SquidFleeGoal`, the burst of speed away from whatever just hit the squid.
//!
//! Ports the inner class of `net.minecraft.world.entity.animal.squid.Squid`. It is the priority-1
//! goal and, unlike the drift it sits beside, it only writes a thrust when the destination block is
//! water or air — a squid pinned against stone keeps whatever thrust it already had rather than
//! pushing into the wall.

use steel_registry::blocks::block_state_ext::BlockStateExt as _;
use steel_utils::BlockPos;

use crate::entity::ai::goal::selector::{Goal, GoalControls};
use crate::entity::{PathfinderMob, SquidMovementVector};
use crate::fluid::FluidStateExt as _;

/// Squared range within which the squid flees, vanilla's `< 100.0`.
const FLEE_RANGE_SQUARED: f64 = 100.0;

/// Base flee speed before the distance falloff, vanilla's local `avoidSpeed = 3.0`.
const FLEE_SPEED: f64 = 3.0;

/// Distance at which the flee speed starts falling off, vanilla's `> 5.0`.
const FLEE_FALLOFF_START: f64 = 5.0;

/// Divisor of the falloff ramp, vanilla's `/ 5.0`.
const FLEE_FALLOFF_SPAN: f64 = 5.0;

/// Divisor turning the flee velocity into a per-tick thrust, vanilla's `/ 20.0`.
const FLEE_THRUST_DIVISOR: f64 = 20.0;

/// Vanilla `Squid.SquidFleeGoal`.
pub struct SquidFleeGoal {
    movement_vector: SquidMovementVector,
    flee_ticks: i32,
}

impl SquidFleeGoal {
    /// Creates the goal over the squid's shared thrust handle.
    #[must_use]
    pub(crate) fn new(movement_vector: &SquidMovementVector) -> Self {
        Self {
            movement_vector: movement_vector.clone(),
            flee_ticks: 0,
        }
    }

    /// Applies vanilla's distance falloff to the flee speed.
    ///
    /// Returns `None` where vanilla's `avoidSpeed > 0.0` guard fails, which happens once the
    /// attacker is more than twenty blocks away — at that point vanilla leaves the direction
    /// unscaled rather than scaling it to nothing.
    fn falloff_speed(distance: f64) -> Option<f64> {
        let speed = if distance > FLEE_FALLOFF_START {
            FLEE_SPEED - (distance - FLEE_FALLOFF_START) / FLEE_FALLOFF_SPAN
        } else {
            FLEE_SPEED
        };

        (speed > 0.0).then_some(speed)
    }
}

impl Goal for SquidFleeGoal {
    /// Vanilla declares no controls, so the drift goal keeps running underneath this one.
    fn controls(&self) -> GoalControls {
        GoalControls::EMPTY
    }

    fn can_use(&mut self, mob: &dyn PathfinderMob) -> bool {
        let Some(attacker) = mob.last_hurt_by_mob() else {
            return false;
        };

        mob.is_in_water()
            && mob.position().distance_squared(attacker.position()) < FLEE_RANGE_SQUARED
    }

    fn start(&mut self, _mob: &dyn PathfinderMob) {
        self.flee_ticks = 0;
    }

    fn requires_update_every_tick(&self) -> bool {
        true
    }

    fn tick(&mut self, mob: &dyn PathfinderMob) {
        self.flee_ticks = self.flee_ticks.wrapping_add(1);

        let Some(attacker) = mob.last_hurt_by_mob() else {
            return;
        };
        let Some(world) = mob.level() else {
            return;
        };

        let position = mob.position();
        let mut flee_to = position - attacker.position();

        // Vanilla samples the block one full flee-vector away, before any normalization, so the
        // sample point moves further out the further the attacker is.
        let sample = BlockPos::containing(
            position.x + flee_to.x,
            position.y + flee_to.y,
            position.z + flee_to.z,
        );
        let state = world.get_block_state(sample);
        let is_air = state.is_air();
        if !state.get_fluid_state().is_water() && !is_air {
            return;
        }

        let length = flee_to.length();
        if length > 0.0 {
            flee_to = flee_to.normalize();
            if let Some(speed) = Self::falloff_speed(length) {
                flee_to *= speed;
            }
        }

        if is_air {
            // A squid breaking the surface loses its vertical thrust rather than launching itself.
            flee_to.y = 0.0;
        }

        self.movement_vector.set(flee_to / FLEE_THRUST_DIVISOR);
    }
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::*;

    #[test]
    fn speed_is_flat_inside_the_falloff_start() {
        assert_eq!(SquidFleeGoal::falloff_speed(0.5), Some(FLEE_SPEED));
        assert_eq!(SquidFleeGoal::falloff_speed(5.0), Some(FLEE_SPEED));
    }

    #[test]
    fn speed_ramps_down_past_the_falloff_start() {
        // Ten blocks out is one full span past the start, so vanilla sheds exactly 1.0.
        assert_eq!(SquidFleeGoal::falloff_speed(10.0), Some(FLEE_SPEED - 1.0));
    }

    #[test]
    fn speed_vanishes_at_twenty_blocks() {
        // 3.0 - (20.0 - 5.0) / 5.0 == 0.0, and vanilla's guard is strictly greater than zero.
        assert_eq!(SquidFleeGoal::falloff_speed(20.0), None);
        assert_eq!(SquidFleeGoal::falloff_speed(25.0), None);
    }

    #[test]
    fn the_goal_shares_the_squids_thrust_handle() {
        let vector = SquidMovementVector::default();
        let goal = SquidFleeGoal::new(&vector);

        goal.movement_vector.set(DVec3::new(0.1, 0.0, 0.1));

        assert_eq!(vector.get(), DVec3::new(0.1, 0.0, 0.1));
    }

    #[test]
    fn a_fresh_goal_starts_its_tick_counter_at_zero() {
        let vector = SquidMovementVector::default();
        let goal = SquidFleeGoal::new(&vector);

        assert_eq!(goal.flee_ticks, 0);
    }
}
