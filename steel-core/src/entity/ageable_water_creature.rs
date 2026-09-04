//! Shared vanilla `AgeableWaterCreature` state and hooks.
//!
//! Ports `net.minecraft.world.entity.AgeableWaterCreature`, the base under squid, glow squid and
//! dolphin. It sits beside [`Animal`](super::Animal) rather than under it: both extend
//! [`AgeableMob`], but a water creature has no breeding, no food and no love state, so it gets its
//! own eighty-line base instead of inheriting one it would have to disable.
//!
//! Vanilla's `WaterAnimal` is a near-copy of this class hanging off `PathfinderMob` instead of
//! `AgeableMob`, for the fish that do not age. The two bodies really are the same, which is why
//! [`spawn_placements::SpawnPredicate`] keeps a separate arm for each and shares the
//! implementation; when Steel gains the fish, this module is where their base belongs too.

use crate::entity::ai::path::PathType;
use crate::entity::damage::DamageSource;
use crate::entity::{AgeableMob, LivingEntity, MobBase};
use steel_registry::vanilla_damage_types;

/// Air ticks a water creature is topped back up to whenever it is in water.
///
/// Vanilla writes the bare `300`, matching the vanilla air supply maximum.
const MAX_AIR_SUPPLY: i32 = 300;

/// Drowning damage a beached water creature takes per tick once its air runs out.
const DROWNING_DAMAGE: f32 = 2.0;

/// Vanilla `AgeableWaterCreature.getAmbientSoundInterval`.
const AMBIENT_SOUND_INTERVAL: i32 = 120;

/// Vanilla `AgeableWaterCreature`, the ageing water mobs' shared behavior.
///
/// Note the air-supply logic is inverted from a land mob's: a water creature suffocates *out* of
/// water, and `handle_air_supply` runs after the base tick with the pre-tick reading so that the
/// base tick's own air handling is overridden rather than compounded.
pub trait AgeableWaterCreature: AgeableMob {
    /// Applies vanilla's `setPathfindingMalus(PathType.WATER, 0.0F)`.
    ///
    /// Water costs a land mob dearly and costs these mobs nothing, which is the whole reason a
    /// squid's random movement stays in the sea.
    fn initialize_water_pathfinding_malus(mob_base: &MobBase)
    where
        Self: Sized,
    {
        mob_base
            .pathfinding_malus()
            .lock()
            .set(PathType::Water, 0.0);
    }

    /// Returns vanilla `AgeableWaterCreature.getAmbientSoundInterval`.
    fn ambient_sound_interval_water_creature(&self) -> i32 {
        AMBIENT_SOUND_INTERVAL
    }

    /// Returns vanilla `AgeableWaterCreature.getBaseExperienceReward`.
    fn base_experience_reward_water_creature(&self) -> i32 {
        1 + rand::random_range(0..3)
    }

    /// Returns vanilla `AgeableWaterCreature.isPushedByFluid`, which is always `false`.
    fn is_pushed_by_fluid_water_creature(&self) -> bool {
        false
    }

    /// Returns vanilla `AgeableWaterCreature.canBeLeashed`, which is `false` before overrides.
    ///
    /// Squid overrides this back to `true`; dolphin and glow squid do not.
    ///
    /// Nothing consults this yet. Steel reaches `can_be_leashed` through the blanket
    /// `impl<T: Mob> Leashable for T`, whose own default answers `true` for every mob and carries a
    /// TODO for hostile mobs, so there is no per-type seam to route this through. Squid's answer is
    /// therefore already correct by coincidence, and this method becomes live the moment
    /// [`Leashable::can_be_leashed`](crate::entity::leash::Leashable::can_be_leashed) gains a `Mob`
    /// hook — at which point dolphin and glow squid start needing it.
    #[expect(
        dead_code,
        reason = "records the vanilla AgeableWaterCreature default until Leashable::can_be_leashed \
                  gains a per-mob hook to route it through"
    )]
    fn can_be_leashed_water_creature(&self) -> bool {
        false
    }

    /// Runs vanilla `AgeableWaterCreature.handleAirSupply`.
    ///
    /// `pre_tick_air_supply` is the reading taken *before* the base tick ran, exactly as vanilla
    /// captures it in `baseTick`. Passing the post-tick value instead would double-count the base
    /// tick's own decrement and drown a beached squid at twice the rate.
    ///
    /// The branch is inverted from a land mob's: air drains *out* of water and refills to full the
    /// moment the mob is back in it, with no gradual recovery, because vanilla assigns the constant
    /// rather than stepping toward it.
    fn handle_air_supply(&self, pre_tick_air_supply: i32) {
        if LivingEntity::is_alive(self) && !self.is_in_water() {
            self.set_air_supply(pre_tick_air_supply - 1);
            if self.should_take_drowning_damage() {
                self.set_air_supply(0);
                if let Some(world) = self.level() {
                    self.hurt(
                        &world,
                        &DamageSource::environment(&vanilla_damage_types::DROWN),
                        DROWNING_DAMAGE,
                    );
                }
            }
        } else {
            self.set_air_supply(MAX_AIR_SUPPLY);
        }
    }
}
