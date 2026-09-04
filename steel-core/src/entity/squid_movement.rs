//! The thrust vector a squid's goals write and its `aiStep` consumes.
//!
//! Vanilla keeps `Squid.movementVector` as a private field and reaches it from
//! `Squid.SquidRandomMovementGoal` and `Squid.SquidFleeGoal` because both are inner classes of
//! `Squid`. Steel's goals are free-standing values owned by the goal selector, so the field becomes
//! a shared handle instead: the entity and both goals hold clones of the same cell.
//!
//! The vector is a *thrust*, not a destination. `Squid.aiStep` assigns it to the entity's velocity
//! only on the power stroke of the tentacle cycle and lets the squid coast between strokes, which is
//! why a squid moves in bursts rather than gliding.

use std::sync::Arc;

use glam::DVec3;
use steel_utils::locks::SyncMutex;

/// Squared length below which vanilla `Squid.hasMovementVector` reports no thrust.
///
/// Vanilla writes `1.0E-5F` and compares it against a `double` length, so the literal widens to
/// `f64` before the comparison; keeping it `f64` here reproduces that without a cast at the site.
const SIGNIFICANT_LENGTH_SQUARED: f64 = 1.0e-5;

/// Shared handle on vanilla `Squid.movementVector`.
///
/// Cloning shares the cell rather than copying the vector, which is the whole point: the entity
/// reads what its goals wrote. Cheap to clone and safe to hold across a goal tick.
#[derive(Clone, Debug, Default)]
pub struct SquidMovementVector(Arc<SyncMutex<DVec3>>);

impl SquidMovementVector {
    /// Returns the current thrust.
    #[must_use]
    pub fn get(&self) -> DVec3 {
        *self.0.lock()
    }

    /// Overwrites the thrust, vanilla's `Squid.this.movementVector = ...`.
    pub fn set(&self, thrust: DVec3) {
        *self.0.lock() = thrust;
    }

    /// Returns vanilla `Squid.hasMovementVector`.
    ///
    /// The random-movement goal re-rolls whenever this is `false`, so an exactly-zero vector is not
    /// a resting state but a request for a fresh direction on the next tick.
    #[must_use]
    pub fn is_significant(&self) -> bool {
        self.get().length_squared() > SIGNIFICANT_LENGTH_SQUARED
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_share_one_cell() {
        let vector = SquidMovementVector::default();
        let goal_handle = vector.clone();

        goal_handle.set(DVec3::new(0.2, -0.1, 0.0));

        assert_eq!(vector.get(), DVec3::new(0.2, -0.1, 0.0));
    }

    #[test]
    fn zero_thrust_is_not_significant() {
        let vector = SquidMovementVector::default();

        assert!(!vector.is_significant());
    }

    #[test]
    fn thrust_below_the_vanilla_threshold_is_not_significant() {
        let vector = SquidMovementVector::default();

        // 1e-3 squared is 1e-6, which is below vanilla's 1e-5 cutoff.
        vector.set(DVec3::new(1.0e-3, 0.0, 0.0));

        assert!(!vector.is_significant());
    }

    #[test]
    fn a_rolled_direction_is_significant() {
        let vector = SquidMovementVector::default();

        vector.set(DVec3::new(0.2, 0.0, 0.0));

        assert!(vector.is_significant());
    }
}
