//! Vanilla Squid entity: the jet-propelled cephalopod that ignores the pathfinder entirely.
//!
//! Ports `net.minecraft.world.entity.animal.Squid`. A squid is the odd one out among Steel's mobs
//! so far: it has no navigation and no move control worth the name. Its two goals write a raw
//! velocity into a shared movement vector, `ai_step` applies that vector on the power stroke of a
//! tentacle cycle, and [`LivingEntity::travel`] is overridden to move by the velocity directly
//! rather than through walking physics. That is why a squid drifts in bursts instead of swimming
//! at a steady speed.
//!
//! The tentacle fields are not decoration. `tentacle_movement` is the phase of the stroke, and the
//! velocity is only applied while that phase is past three quarters of its first half — so the
//! shared vector is a *thrust*, not a target, and the squid coasts between strokes at `0.9` drag.

use std::sync::{Arc, Weak};

use glam::DVec3;
use simdnbt::borrow::NbtCompound as BorrowedNbtCompoundView;
use simdnbt::owned::NbtCompound;
use steel_macros::entity_behavior;
use steel_protocol::packets::game::SoundSource;
use steel_registry::entity_data::EntityPose;
use steel_registry::entity_type::{EntityDimensions, EntityTypeRef};
use steel_registry::particle_type::{ParticleData, ParticleTypeRef};
use steel_registry::sound_event::SoundEventRef;
use steel_registry::vanilla_entity_data::SquidEntityData;
use steel_registry::{sound_events, vanilla_particle_types};
use steel_utils::entity_events::EntityStatus;
use steel_utils::locks::SyncMutex;
use steel_utils::{BlockPos, BlockStateId, DowncastType, DowncastTypeKey};

use crate::entity::ai::goal::{SquidFleeGoal, SquidRandomMovementGoal};
use crate::entity::damage::DamageSource;
use crate::entity::{
    AgeableMob, AgeableMobBase, AgeableMobGroupData, AgeableWaterCreature, Entity, EntityBase,
    EntityBaseLoad, EntityEventSource as _, EntitySpawnReason, EntitySyncedData, LivingEntity,
    LivingEntityBase, Mob, MobBase, PathfinderMob, SpawnGroupData, SquidMovementVector,
};
use crate::physics::{MoveResult, MoverType};
use crate::world::World;

/// Vanilla's `AgeableMob.AgeableMobGroupData(0.05F)` baby chance for a squid shoal.
const SQUID_BABY_SPAWN_CHANCE: f32 = 0.05;

/// Vanilla `Squid.getDefaultGravity`.
const SQUID_GRAVITY: f64 = 0.08;

/// Drag applied to the coasting half of the tentacle cycle, vanilla's `scale(0.9)`.
const COAST_DRAG: f64 = 0.9;

/// Rotation-speed decay while thrusting, vanilla's `rotateSpeed *= 0.8F`.
const THRUST_ROTATION_DECAY: f32 = 0.8;

/// Rotation-speed decay while coasting, vanilla's `rotateSpeed *= 0.99F`.
const COAST_ROTATION_DECAY: f32 = 0.99;

/// Fraction of the power half-stroke after which the thrust fires, vanilla's `> 0.75`.
const THRUST_PHASE_THRESHOLD: f32 = 0.75;

/// Chance denominator for re-rolling the tentacle speed each cycle, vanilla's `nextInt(10) == 0`.
const TENTACLE_SPEED_REROLL_CHANCE: i32 = 10;

/// Smoothing applied to body yaw and pitch each tick, vanilla's `* 0.1F`.
const BODY_ROTATION_SMOOTHING: f32 = 0.1;

/// Pitch a beached squid settles to, vanilla's `-90.0F` target.
const BEACHED_PITCH: f32 = -90.0;

/// Smoothing applied to that settle, vanilla's `* 0.02F`.
const BEACHED_PITCH_SMOOTHING: f32 = 0.02;

/// Vanilla `Squid.zBodyRot` advance per tick, scaled by the rotation speed.
const Z_BODY_ROTATION_SCALE: f32 = 1.5;

const SQUID_BABY_WIDTH: f32 = 0.5;
const SQUID_BABY_HEIGHT: f32 = 0.5;
const SQUID_BABY_EYE_HEIGHT: f32 = 0.37;

const SQUID_BABY_DIMENSIONS: EntityDimensions =
    EntityDimensions::new(SQUID_BABY_WIDTH, SQUID_BABY_HEIGHT, SQUID_BABY_EYE_HEIGHT);

/// Ink particles one squirt sends, vanilla's `for (int i = 0; i < 30; i++)`.
const INK_PARTICLE_COUNT: i32 = 30;

/// Width of the horizontal ink scatter, vanilla's `nextFloat() * 0.6`.
const INK_SPREAD_SPAN: f64 = 0.6;

/// Half of [`INK_SPREAD_SPAN`], subtracted to center the scatter, vanilla's `- 0.3`.
const INK_SPREAD_OFFSET: f64 = 0.3;

/// Ink offset scale for a baby squid, vanilla's `isBaby() ? 0.1F`.
const BABY_INK_OFFSET_SCALE: f64 = 0.1;

/// Ink offset scale for an adult squid, vanilla's `: 0.3F`.
const ADULT_INK_OFFSET_SCALE: f64 = 0.3;

/// Span of the random ink offset stretch, vanilla's `+ nextFloat() * 2.0F`.
const INK_OFFSET_RANDOM_SPAN: f64 = 2.0;

/// Height above the ink origin each particle spawns at, vanilla's `pos.y + 0.5`.
const INK_ORIGIN_Y_OFFSET: f64 = 0.5;

/// Particle speed vanilla passes for ink, its `0.1F` trailing argument.
const INK_PARTICLE_SPEED: f64 = 0.1;

/// Rotates a vector about the X axis, vanilla `Vec3.xRot`.
fn rotate_x(vector: DVec3, radians: f32) -> DVec3 {
    let cos = f64::from(radians.cos());
    let sin = f64::from(radians.sin());
    DVec3::new(
        vector.x,
        vector.y * cos + vector.z * sin,
        vector.z * cos - vector.y * sin,
    )
}

/// Rotates a vector about the Y axis, vanilla `Vec3.yRot`.
fn rotate_y(vector: DVec3, radians: f32) -> DVec3 {
    let cos = f64::from(radians.cos());
    let sin = f64::from(radians.sin());
    DVec3::new(
        vector.x * cos + vector.z * sin,
        vector.y,
        vector.z * cos - vector.x * sin,
    )
}

/// Client-facing tentacle and body-rotation state, kept together because every field moves in step.
#[derive(Debug, Clone, Copy)]
struct SquidAnimationState {
    /// Phase of the current tentacle stroke, in radians.
    tentacle_movement: f32,
    /// Previous tick's stroke phase.
    old_tentacle_movement: f32,
    /// Current tentacle spread angle.
    tentacle_angle: f32,
    /// Previous tick's spread angle.
    old_tentacle_angle: f32,
    /// Radians per tick the stroke advances; re-rolled roughly once per ten cycles.
    tentacle_speed: f32,
    /// How fast the body is spinning about its own axis.
    rotate_speed: f32,
    /// Body pitch.
    x_body_rot: f32,
    /// Previous tick's body pitch, which the ink cloud is oriented by.
    x_body_rot_o: f32,
    /// Body roll.
    z_body_rot: f32,
    /// Previous tick's body roll.
    z_body_rot_o: f32,
}

impl SquidAnimationState {
    /// Vanilla's constructor draw: `1.0F / (random.nextFloat() + 1.0F) * 0.2F`.
    fn new() -> Self {
        Self {
            tentacle_movement: 0.0,
            old_tentacle_movement: 0.0,
            tentacle_angle: 0.0,
            old_tentacle_angle: 0.0,
            tentacle_speed: Self::roll_tentacle_speed(),
            rotate_speed: 0.0,
            x_body_rot: 0.0,
            x_body_rot_o: 0.0,
            z_body_rot: 0.0,
            z_body_rot_o: 0.0,
        }
    }

    fn roll_tentacle_speed() -> f32 {
        1.0 / (rand::random::<f32>() + 1.0) * 0.2
    }
}

#[entity_behavior(class = "Squid")]
/// Vanilla squid entity.
pub struct SquidEntity {
    base: EntityBase,
    entity_type: EntityTypeRef,
    living_base: LivingEntityBase,
    mob_base: MobBase,
    ageable_base: AgeableMobBase,
    /// The thrust its goals write and `ai_step` consumes, shared with both goals.
    movement_vector: SquidMovementVector,
    animation: SyncMutex<SquidAnimationState>,
    entity_data: SyncMutex<SquidEntityData>,
}

// SAFETY: This key is owned by Steel and uniquely identifies `SquidEntity`.
unsafe impl DowncastType for SquidEntity {
    const TYPE_KEY: DowncastTypeKey = DowncastTypeKey::new("steel:entity/squid");
}

impl SquidEntity {
    /// Creates a new squid at runtime.
    #[must_use]
    pub fn new(entity_type: EntityTypeRef, id: i32, position: DVec3, world: Weak<World>) -> Self {
        Self::new_with_base(
            EntityBase::new(id, position, entity_type.dimensions, world),
            entity_type,
        )
    }

    /// Reconstructs a squid from persisted base entity state.
    #[must_use]
    pub fn from_saved(entity_type: EntityTypeRef, load: EntityBaseLoad) -> Self {
        Self::new_with_base(
            EntityBase::from_load(load, entity_type.dimensions),
            entity_type,
        )
    }

    fn new_with_base(base: EntityBase, entity_type: EntityTypeRef) -> Self {
        let living_base = LivingEntityBase::new(entity_type);
        let mob_base = MobBase::new();
        let ageable_base = AgeableMobBase::new();
        <Self as AgeableWaterCreature>::initialize_water_pathfinding_malus(&mob_base);
        let mut entity_data = SquidEntityData::new();
        living_base.initialize_synced_data(&mut entity_data);

        // Both goals write the same vector the entity reads, so the handle is created here and
        // shared three ways. Vanilla gets this for free by making the goals inner classes.
        let movement_vector = SquidMovementVector::default();

        {
            let mut goal_selector = mob_base.goal_selector().lock();
            goal_selector.add_goal(0, SquidRandomMovementGoal::new(&movement_vector));
            goal_selector.add_goal(1, SquidFleeGoal::new(&movement_vector));
        }

        Self {
            base,
            entity_type,
            living_base,
            mob_base,
            ageable_base,
            movement_vector,
            animation: SyncMutex::new(SquidAnimationState::new()),
            entity_data: SyncMutex::new(entity_data),
        }
    }

    /// Returns vanilla `Squid.hasMovementVector`.
    #[must_use]
    pub fn has_movement_vector(&self) -> bool {
        self.movement_vector.is_significant()
    }

    /// Returns the squirt sound this squid makes when it inks, vanilla `Squid.getSquirtSound`.
    fn squirt_sound(&self) -> SoundEventRef {
        &sound_events::ENTITY_SQUID_SQUIRT
    }

    /// Returns the ink particle this squid squirts, vanilla `Squid.getInkParticle`.
    ///
    /// Glow squid overrides it; the base squid uses plain ink.
    fn ink_particle(&self) -> ParticleTypeRef {
        &vanilla_particle_types::SQUID_INK
    }

    /// Rotates a body-local vector into world space, vanilla `Squid.rotateVector`.
    ///
    /// Both rotations read the *previous* tick's body angles, which is vanilla's own choice: the ink
    /// leaves from where the squid was pointing when it was hit, not from where the same tick's
    /// rotation smoothing has since carried it.
    fn rotate_vector(&self, vector: DVec3) -> DVec3 {
        let x_body_rot_o = self.animation.lock().x_body_rot_o;
        let pitched = rotate_x(vector, x_body_rot_o.to_radians());
        let y_body_rot_o = LivingEntity::living_rotation_state(self).y_body_rot_o();
        rotate_y(pitched, -y_body_rot_o.to_radians())
    }

    /// Runs vanilla `Squid.spawnInk`.
    ///
    /// The cloud is thirty particles fired straight down in body space, so a squid inks toward its
    /// own belly rather than toward the attacker — tilt the squid and the cloud tilts with it.
    fn spawn_ink(&self, world: &World) {
        LivingEntity::make_sound(self, Some(self.squirt_sound()));

        let origin = self.rotate_vector(DVec3::new(0.0, -1.0, 0.0)) + self.position();
        let offset_scale = if AgeableMob::is_baby(self) {
            BABY_INK_OFFSET_SCALE
        } else {
            ADULT_INK_OFFSET_SCALE
        };

        for _ in 0..INK_PARTICLE_COUNT {
            let direction = self.rotate_vector(DVec3::new(
                f64::from(rand::random::<f32>()) * INK_SPREAD_SPAN - INK_SPREAD_OFFSET,
                -1.0,
                f64::from(rand::random::<f32>()) * INK_SPREAD_SPAN - INK_SPREAD_OFFSET,
            ));
            let spread = direction
                * (offset_scale + f64::from(rand::random::<f32>()) * INK_OFFSET_RANDOM_SPAN);

            // Vanilla sends each particle as its own `count = 0` packet, which makes the client read
            // the offset triple as a velocity rather than as a spread box.
            world.send_particles(
                ParticleData::simple(self.ink_particle()),
                DVec3::new(origin.x, origin.y + INK_ORIGIN_Y_OFFSET, origin.z),
                0,
                spread,
                INK_PARTICLE_SPEED,
            );
        }
    }

    fn update_dirty_mob_effect_entity_data(&self) {
        if !self.living_base.take_effects_dirty() {
            return;
        }

        let display = self.living_base.mob_effect_display_state();

        {
            let mut entity_data = self.entity_data.lock();
            let living = entity_data.living_entity_mut();
            living.effect_particles.set(display.particles);
            living.effect_ambience.set(display.ambient);
        }

        self.entity_data.set_base_invisible_flag(display.invisible);
        self.entity_data
            .set_base_glowing_flag(self.has_glowing_tag() || display.glowing);
    }

    /// Advances the tentacle stroke, returning whether the cycle wrapped this tick.
    ///
    /// Vanilla wraps by subtracting a full turn on the server and clamping on the client, then
    /// broadcasts `SQUID_ANIM_SYNCH` so both agree on the phase.
    fn advance_tentacle_cycle(&self) -> bool {
        let mut animation = self.animation.lock();
        animation.old_tentacle_movement = animation.tentacle_movement;
        animation.old_tentacle_angle = animation.tentacle_angle;
        animation.x_body_rot_o = animation.x_body_rot;
        animation.z_body_rot_o = animation.z_body_rot;
        animation.tentacle_movement += animation.tentacle_speed;

        if animation.tentacle_movement <= std::f32::consts::TAU {
            return false;
        }

        animation.tentacle_movement -= std::f32::consts::TAU;
        if rand::random_range(0..TENTACLE_SPEED_REROLL_CHANCE) == 0 {
            animation.tentacle_speed = SquidAnimationState::roll_tentacle_speed();
        }
        true
    }

    /// Runs the in-water half of vanilla `Squid.aiStep`.
    fn swim_step(&self) {
        let thrust = {
            let mut animation = self.animation.lock();
            if animation.tentacle_movement < std::f32::consts::PI {
                let phase = animation.tentacle_movement / std::f32::consts::PI;
                animation.tentacle_angle =
                    (phase * phase * std::f32::consts::PI).sin() * std::f32::consts::PI * 0.25;
                if phase > THRUST_PHASE_THRESHOLD {
                    animation.rotate_speed = 1.0;
                    Some(self.movement_vector.get())
                } else {
                    animation.rotate_speed *= THRUST_ROTATION_DECAY;
                    None
                }
            } else {
                animation.tentacle_angle = 0.0;
                animation.rotate_speed *= COAST_ROTATION_DECAY;
                None
            }
        };

        match thrust {
            Some(vector) => self.set_velocity(vector),
            None if self.animation.lock().tentacle_movement >= std::f32::consts::PI => {
                self.set_velocity(self.velocity() * COAST_DRAG);
            }
            None => {}
        }

        let movement = self.velocity();
        let horizontal = movement.x.hypot(movement.z);
        let mut animation = self.animation.lock();
        let target_yaw = -(movement.x.atan2(movement.z) as f32).to_degrees();
        let y_body_rot = self.living_base.y_body_rot();
        let new_yaw = y_body_rot + (target_yaw - y_body_rot) * BODY_ROTATION_SMOOTHING;
        drop(animation);
        self.living_base.set_y_body_rot(new_yaw);
        self.set_rotation((new_yaw, self.rotation().1));

        animation = self.animation.lock();
        animation.z_body_rot +=
            std::f32::consts::PI * animation.rotate_speed * Z_BODY_ROTATION_SCALE;
        let target_pitch = -(horizontal.atan2(movement.y) as f32).to_degrees();
        animation.x_body_rot += (target_pitch - animation.x_body_rot) * BODY_ROTATION_SMOOTHING;
    }

    /// Runs the out-of-water half of vanilla `Squid.aiStep`.
    fn beached_step(&self) {
        {
            let mut animation = self.animation.lock();
            animation.tentacle_angle =
                animation.tentacle_movement.sin().abs() * std::f32::consts::PI * 0.25;
        }

        let vertical = self.velocity().y - self.get_gravity();
        self.set_velocity(DVec3::new(0.0, vertical * COAST_DRAG, 0.0));

        let mut animation = self.animation.lock();
        animation.x_body_rot += (BEACHED_PITCH - animation.x_body_rot) * BEACHED_PITCH_SMOOTHING;
    }
}

impl Entity for SquidEntity {
    fn base(&self) -> &EntityBase {
        &self.base
    }

    fn entity_type(&self) -> EntityTypeRef {
        self.entity_type
    }

    /// Vanilla `AgeableWaterCreature.baseTick`, which reads the air supply before the base tick.
    fn base_tick(&self) {
        let pre_tick_air_supply = self.air_supply();
        Mob::base_tick_mob(self);
        self.handle_air_supply(pre_tick_air_supply);
    }

    fn dimensions_for_pose(&self, _pose: EntityPose) -> EntityDimensions {
        let scale = LivingEntity::get_scale(self);
        if AgeableMob::is_baby(self) {
            SQUID_BABY_DIMENSIONS.scale(scale)
        } else if self.entity_type.fixed {
            self.entity_type.dimensions
        } else {
            self.entity_type.dimensions.scale(scale)
        }
    }

    fn synced_data(&self) -> Option<&dyn EntitySyncedData> {
        Some(&self.entity_data)
    }

    fn update_data_before_sync(&self) {
        self.update_dirty_mob_effect_entity_data();
    }

    fn get_gravity(&self) -> f64 {
        SQUID_GRAVITY
    }

    /// Vanilla `AgeableWaterCreature.isPushedByFluid`.
    fn is_pushed_by_fluid(&self) -> bool {
        self.is_pushed_by_fluid_water_creature()
    }

    fn sound_source(&self) -> SoundSource {
        SoundSource::Neutral
    }

    fn play_step_sound(&self, _pos: BlockPos, _block_state: BlockStateId) {}

    fn save_additional(&self, nbt: &mut NbtCompound) {
        self.save_mob(nbt);
        self.save_ageable_mob(nbt);
    }

    fn load_additional(&self, nbt: BorrowedNbtCompoundView<'_, '_>) {
        self.load_mob(nbt);
        self.load_ageable_mob(nbt);
    }
}

impl LivingEntity for SquidEntity {
    fn living_base(&self) -> &LivingEntityBase {
        &self.living_base
    }

    fn get_health(&self) -> f32 {
        *self.entity_data.lock().living_entity().health.get()
    }

    fn set_health(&self, health: f32) {
        let max_health = self.get_max_health();
        let clamped = health.clamp(0.0, max_health);
        self.entity_data
            .lock()
            .living_entity_mut()
            .health
            .set(clamped);
    }

    fn sound_volume(&self) -> f32 {
        0.4
    }

    fn hurt_sound(&self, _source: &DamageSource) -> Option<SoundEventRef> {
        Some(&sound_events::ENTITY_SQUID_HURT)
    }

    fn death_sound(&self) -> Option<SoundEventRef> {
        Some(&sound_events::ENTITY_SQUID_DEATH)
    }

    /// Vanilla `AgeableWaterCreature.getBaseExperienceReward`, which is `1 + random(3)`.
    fn base_experience_reward(&self) -> i32 {
        self.base_experience_reward_water_creature()
    }

    /// Vanilla `Squid.hurtServer`, which inks only when the damage came from a mob.
    ///
    /// The `last_hurt_by_mob` read happens *after* the base hurt has run, because that is what sets
    /// it — an environmental death (drowning on land, suffocation) therefore leaves no ink.
    fn hurt_server(&self, world: &World, source: &DamageSource, amount: f32) -> bool {
        if !self.hurt_server_living_entity(world, source, amount) {
            return false;
        }

        if self.last_hurt_by_mob().is_some() {
            self.spawn_ink(world);
        }

        true
    }

    fn server_ai_step(&self) {
        Mob::mob_server_ai_step(self);
    }

    /// Vanilla `Squid.travel`, which discards the walking input entirely.
    ///
    /// A squid is moved only by the velocity its stroke wrote, so `travel` skips the whole
    /// air/water/fall-flying ladder and moves by the current velocity.
    fn travel(&self, _input: DVec3) -> Option<MoveResult> {
        self.move_entity(MoverType::SelfMovement, self.velocity())
    }

    fn ai_step(&self) -> Option<MoveResult> {
        let result = self.default_ai_step();

        AgeableMob::tick_ageable_mob(self);

        if self.advance_tentacle_cycle() {
            self.broadcast_entity_event(EntityStatus::SquidAnimSynch);
        }

        if self.is_in_water() {
            self.swim_step();
        } else {
            self.beached_step();
        }

        result
    }
}

impl AgeableMob for SquidEntity {
    fn ageable_base(&self) -> &AgeableMobBase {
        &self.ageable_base
    }

    fn is_age_locked(&self) -> bool {
        *self.entity_data.lock().ageable_mob().age_locked.get()
    }

    fn set_age_locked(&self, age_locked: bool) {
        self.entity_data
            .lock()
            .ageable_mob_mut()
            .age_locked
            .set(age_locked);
    }

    fn set_synced_baby(&self, baby: bool) {
        self.entity_data.lock().ageable_mob_mut().baby.set(baby);
    }

    fn age_boundary_changed(&self, _baby: bool) {
        self.refresh_dimensions();
    }
}

impl AgeableWaterCreature for SquidEntity {
    /// Vanilla `Squid.canBeLeashed`, which overrides the water-creature `false` back to `true`.
    fn can_be_leashed_water_creature(&self) -> bool {
        true
    }
}

impl Mob for SquidEntity {
    fn mob_base(&self) -> &MobBase {
        &self.mob_base
    }

    fn tick_goal_selectors(&self) {
        PathfinderMob::tick_pathfinder_goal_selectors(self);
    }

    fn tick_path_navigation(&self) {
        PathfinderMob::tick_pathfinder_path_navigation(self);
    }

    fn ambient_sound(&self) -> Option<SoundEventRef> {
        Some(&sound_events::ENTITY_SQUID_AMBIENT)
    }

    fn ambient_sound_interval(&self) -> i32 {
        self.ambient_sound_interval_water_creature()
    }

    /// Vanilla `AgeableWaterCreature.checkSpawnObstruction`, which drops the fluid test.
    ///
    /// A land mob refuses to spawn in liquid; a squid requires it, so the base class overrides the
    /// check down to the entity-overlap half alone.
    fn check_spawn_obstruction(&self, world: &Arc<World>) -> bool {
        let bounding_box = self.bounding_box();
        let this = self.as_entity_event_source();
        !world.has_entity_in_aabb_matching(&bounding_box, |other| {
            other.id() != this.id()
                && !other.is_spectator()
                && !other.is_removed()
                && other.blocks_building()
                && !other.is_passenger_of_same_vehicle(this)
        })
    }

    fn finalize_spawn(
        &self,
        world: &Arc<World>,
        spawn_reason: EntitySpawnReason,
        group_data: Option<SpawnGroupData>,
    ) -> Option<SpawnGroupData> {
        let group_data = group_data.or(Some(SpawnGroupData::AgeableMob(
            AgeableMobGroupData::with_baby_spawn_chance(SQUID_BABY_SPAWN_CHANCE),
        )));

        self.finalize_spawn_ageable_mob(world, spawn_reason, group_data)
    }

    fn mob_flags(&self) -> i8 {
        *self.entity_data.lock().mob().mob_flags.get()
    }

    fn set_mob_flags(&self, flags: i8) {
        self.entity_data.lock().mob_mut().mob_flags.set(flags);
    }
}

impl PathfinderMob for SquidEntity {}

#[cfg(test)]
mod tests;
