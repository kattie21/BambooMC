//! Vanilla's `PotentialCalculator`, the crowding field natural spawning charges.
//!
//! Ports `net.minecraft.world.level.PotentialCalculator`. Every mob that counts toward the mob cap
//! contributes a point charge at its position, and a candidate spawn is refused when adding its own
//! charge would push the field's energy past the budget its biome declares. The electrostatic
//! framing is vanilla's own and the physics is only an analogy: this is how vanilla thins spawns
//! near mobs that already exist without giving any mob an explicit exclusion radius.

use super::BlockPos;

/// One mob's contribution to the crowding field.
struct PointCharge {
    /// Block position the charge sits at.
    pos: BlockPos,
    /// Strength of the charge, from the biome's `spawn_costs` entry for the mob's type.
    charge: f64,
}

impl PointCharge {
    /// Vanilla `PointCharge.getPotentialChange`: this charge divided by its distance to `pos`.
    ///
    /// Vanilla measures low corner to low corner (`Vec3i.distSqr`), not center to center, and
    /// returns infinity when the two positions coincide, so no budget can ever accept a second mob
    /// on a block that already holds one.
    fn potential_change(&self, pos: BlockPos) -> f64 {
        let dist_sqr = self.pos.0.as_dvec3().distance_squared(pos.0.as_dvec3());

        if dist_sqr == 0.0 {
            f64::INFINITY
        } else {
            self.charge / dist_sqr.sqrt()
        }
    }
}

/// Vanilla `PotentialCalculator`, a bag of point charges queried as a field.
#[derive(Default)]
pub(crate) struct PotentialCalculator {
    /// Every charge added so far, in insertion order. Vanilla never removes one: the calculator
    /// lives for a single spawn tick and is rebuilt from the live mobs on the next.
    charges: Vec<PointCharge>,
}

impl PotentialCalculator {
    /// Vanilla `addCharge`, which drops a zero charge rather than storing it.
    ///
    /// The skip is load-bearing rather than an optimization: a stored zero charge would still
    /// report infinity for a spawn attempt on its own block, because
    /// [`PointCharge::potential_change`] tests the distance before it looks at the charge.
    pub(crate) fn add_charge(&mut self, pos: BlockPos, charge: f64) {
        if charge != 0.0 {
            self.charges.push(PointCharge { pos, charge });
        }
    }

    /// Vanilla `getPotentialEnergyChange`, the energy a new `charge` at `pos` would add.
    ///
    /// The field's summed potential at `pos`, scaled by the incoming charge. A zero incoming charge
    /// returns zero without summing, which is also what keeps `0.0 * INFINITY` from reaching a
    /// caller as `NaN`.
    pub(crate) fn potential_energy_change(&self, pos: BlockPos, charge: f64) -> f64 {
        if charge == 0.0 {
            return 0.0;
        }

        let potential_change: f64 = self
            .charges
            .iter()
            .map(|point| point.potential_change(pos))
            .sum();

        potential_change * charge
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockPos, PotentialCalculator};

    /// Tolerance for the exactly-representable quotients these tests construct.
    const TOLERANCE: f64 = 1e-12;

    /// Asserts `value` is `expected`, which every quotient below is exactly.
    fn assert_close(value: f64, expected: f64) {
        assert!(
            (value - expected).abs() < TOLERANCE,
            "expected {expected}, got {value}"
        );
    }

    /// A zero charge is dropped, so it cannot make its own block infinitely expensive.
    #[test]
    fn a_zero_charge_is_never_stored() {
        let mut calculator = PotentialCalculator::default();
        let pos = BlockPos::new(4, 64, -7);

        calculator.add_charge(pos, 0.0);

        assert_eq!(calculator.potential_energy_change(pos, 1.0), 0.0);
    }

    /// A zero incoming charge returns zero rather than multiplying infinity into `NaN`.
    #[test]
    fn a_zero_incoming_charge_short_circuits() {
        let mut calculator = PotentialCalculator::default();
        let pos = BlockPos::new(0, 64, 0);
        calculator.add_charge(pos, 5.0);

        let energy = calculator.potential_energy_change(pos, 0.0);

        assert_eq!(energy, 0.0);
        assert!(!energy.is_nan());
    }

    /// The summed potential is scaled by the incoming charge, vanilla's `* charge`.
    #[test]
    fn the_potential_is_scaled_by_the_incoming_charge() {
        let mut calculator = PotentialCalculator::default();
        calculator.add_charge(BlockPos::new(2, 64, 0), 1.0);

        // 1.0 / 2 blocks = 0.5 of potential, times an incoming charge of 3.
        assert_close(
            calculator.potential_energy_change(BlockPos::new(0, 64, 0), 3.0),
            1.5,
        );
    }

    /// Every charge in the field contributes, and distance is measured in three dimensions.
    #[test]
    fn charges_sum_over_the_whole_field() {
        let mut calculator = PotentialCalculator::default();
        calculator.add_charge(BlockPos::new(2, 64, 0), 1.0);
        calculator.add_charge(BlockPos::new(0, 68, 0), 2.0);

        // 1.0/2 from two blocks away, plus 2.0/4 from four blocks up: unlike the block-distance
        // player filter, this one does not drop the vertical component.
        assert_close(
            calculator.potential_energy_change(BlockPos::new(0, 64, 0), 1.0),
            1.0,
        );
    }

    /// A charge on the queried block costs infinity, so no budget accepts the spawn.
    #[test]
    fn a_charge_on_the_queried_block_is_infinitely_expensive() {
        let mut calculator = PotentialCalculator::default();
        let pos = BlockPos::new(-13, 70, 21);
        calculator.add_charge(pos, 0.7);

        let energy = calculator.potential_energy_change(pos, 0.7);

        assert!(energy.is_infinite());
        assert!(energy.is_sign_positive());
    }
}
