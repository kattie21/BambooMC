//! Vanilla `Squid.SquidRandomMovementGoal`, the squid's default drift.
//!
//! Ports the static inner class of `net.minecraft.world.entity.animal.squid.Squid`. It is the
//! priority-0 goal and its `canUse` is an unconditional `true`, so it runs whenever the flee goal is
//! not holding the same controls — a squid is therefore never idle, it is always either drifting or
//! fleeing.

use glam::DVec3;

use crate::entity::ai::goal::reduced_tick_delay;
use crate::entity::ai::goal::selector::{Goal, GoalControls};
use crate::entity::{PathfinderMob, SquidMovementVector};

/// Ticks of stillness after which the squid stops thrusting entirely, vanilla's `> 100`.
///
/// `noActionTime` climbs while no player is nearby, so an unwatched squid parks itself rather than
/// drifting forever — this is vanilla's own idle throttle, not a Steel optimization.
const NO_ACTION_TIME_LIMIT: i32 = 100;

/// Mean interval between direction re-rolls, vanilla's `nextInt(reducedTickDelay(50))`.
const REROLL_INTERVAL_TICKS: i32 = 50;

/// Horizontal thrust magnitude, vanilla's `* 0.2F` on both cosine and sine.
const HORIZONTAL_THRUST: f64 = 0.2;

/// Smallest vertical thrust, vanilla's `-0.1F` before the random addition.
const VERTICAL_THRUST_BASE: f64 = -0.1;

/// Span of the vertical thrust draw, vanilla's `+ nextFloat() * 0.2F`.
const VERTICAL_THRUST_SPAN: f64 = 0.2;

/// Vanilla `Squid.SquidRandomMovementGoal`.
pub struct SquidRandomMovementGoal {
    movement_vector: SquidMovementVector,
}

impl SquidRandomMovementGoal {
    /// Creates the goal over the squid's shared thrust handle.
    #[must_use]
    pub(crate) fn new(movement_vector: &SquidMovementVector) -> Self {
        Self {
            movement_vector: movement_vector.clone(),
        }
    }

    /// Draws vanilla's fresh direction: a uniform yaw with a slight downward bias.
    fn roll_thrust() -> DVec3 {
        let angle = rand::random::<f32>() * core::f32::consts::TAU;
        DVec3::new(
            f64::from(angle.cos()) * HORIZONTAL_THRUST,
            VERTICAL_THRUST_BASE + f64::from(rand::random::<f32>()) * VERTICAL_THRUST_SPAN,
            f64::from(angle.sin()) * HORIZONTAL_THRUST,
        )
    }
}

impl Goal for SquidRandomMovementGoal {
    /// Vanilla declares no controls for this goal, so it never blocks the flee goal.
    ///
    /// Both goals write the same vector and neither uses the navigator, so vanilla lets them
    /// coexist; the flee goal simply overwrites the drift on the ticks it runs.
    fn controls(&self) -> GoalControls {
        GoalControls::EMPTY
    }

    fn can_use(&mut self, _mob: &dyn PathfinderMob) -> bool {
        true
    }

    fn tick(&mut self, mob: &dyn PathfinderMob) {
        if mob.no_action_time() > NO_ACTION_TIME_LIMIT {
            self.movement_vector.set(DVec3::ZERO);
            return;
        }

        // Vanilla's `wasTouchingWater` is the previous fluid sweep's reading, which on the server is
        // the same contact state `is_in_water` reads this tick — the sweep runs before `aiStep`.
        let due = rand::random_range(0..reduced_tick_delay(REROLL_INTERVAL_TICKS).max(1)) == 0;
        if due || !mob.is_in_water() || !self.movement_vector.is_significant() {
            self.movement_vector.set(Self::roll_thrust());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rolled_thrust_stays_inside_vanillas_envelope() {
        // The angle's cosine and sine are drawn in `f32`, as vanilla's `Mth.cos`/`Mth.sin` are, so
        // the magnitude carries single-precision error rather than double-precision error.
        const MAGNITUDE_TOLERANCE: f64 = 1.0e-6;

        for _ in 0..256 {
            let thrust = SquidRandomMovementGoal::roll_thrust();

            let horizontal = thrust.x.hypot(thrust.z);
            assert!(
                (horizontal - HORIZONTAL_THRUST).abs() < MAGNITUDE_TOLERANCE,
                "horizontal thrust should sit on the 0.2 circle, got {horizontal}"
            );
            assert!(thrust.y >= VERTICAL_THRUST_BASE);
            assert!(thrust.y <= VERTICAL_THRUST_BASE + VERTICAL_THRUST_SPAN);
        }
    }

    #[test]
    fn a_rolled_thrust_is_always_significant() {
        // The horizontal magnitude alone is 0.2, far above the 1e-5 cutoff, so a fresh roll can
        // never leave the squid in the "no thrust" state that triggers another re-roll.
        let vector = SquidMovementVector::default();

        vector.set(SquidRandomMovementGoal::roll_thrust());

        assert!(vector.is_significant());
    }

    #[test]
    fn the_goal_shares_the_squids_thrust_handle() {
        let vector = SquidMovementVector::default();
        let goal = SquidRandomMovementGoal::new(&vector);

        goal.movement_vector.set(DVec3::new(0.0, 0.5, 0.0));

        assert_eq!(vector.get(), DVec3::new(0.0, 0.5, 0.0));
    }
}
