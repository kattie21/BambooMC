//! Vanilla Ocelot entity: the untamed jungle feline, its trust mechanic and its three gaits.
//!
//! Ports `net.minecraft.world.entity.animal.feline.Ocelot`. An ocelot is not a cat — it never
//! becomes a pet. It has a single boolean, `trusting`, that a player earns by feeding it fish, and
//! all trust buys is that the ocelot stops fleeing. It keeps its own AI either way.
//!
//! The three speed modifiers are the interesting part of the class and they are shared with
//! [`OcelotAttackGoal`]: whichever goal is driving, `custom_server_ai_step` reads the speed back
//! off the move control and picks the matching pose, so a stalking ocelot crouches and a charging
//! one sprints without either goal ever setting a pose itself.

use std::sync::{Arc, Weak};

use glam::DVec3;
use simdnbt::borrow::NbtCompound as BorrowedNbtCompoundView;
use simdnbt::owned::NbtCompound;
use steel_macros::entity_behavior;
use steel_protocol::packets::game::SoundSource;
use steel_registry::entity_data::EntityPose;
use steel_registry::entity_type::{
    EntityAttachmentPoint, EntityAttachments, EntityDimensions, EntityTypeRef,
};
use steel_registry::item_stack::ItemStack;
use steel_registry::sound_event::SoundEventRef;
use steel_registry::vanilla_entity_data::OcelotEntityData;
use steel_registry::vanilla_item_tags::ItemTag;
use steel_registry::{REGISTRY, TaggedRegistryExt, sound_events, vanilla_attributes};
use steel_utils::entity_events::EntityStatus;
use steel_utils::locks::SyncMutex;
use steel_utils::types::InteractionHand;
use steel_utils::{BlockPos, BlockStateId, DowncastType, DowncastTypeKey};

use crate::behavior::InteractionResult;
use crate::entity::ai::control::MoveControlOperation;
use crate::entity::ai::goal::{
    AvoidEntityGoal, BreedGoal, FloatGoal, LeapAtTargetGoal, LookAtPlayerGoal, OcelotAttackGoal,
    TemptGoal, WaterAvoidingRandomStrollGoal,
};
use crate::entity::damage::DamageSource;
use crate::entity::{
    AgeableMob, AgeableMobBase, AgeableMobGroupData, Animal, AnimalBase, Entity, EntityBase,
    EntityBaseLoad, EntitySpawnReason, EntitySyncedData, LivingEntity, LivingEntityBase, Mob,
    MobBase, PathfinderMob, SpawnGroupData,
};
use crate::physics::MoveResult;
use crate::player::Player;
use crate::world::World;

/// Vanilla `Ocelot::CROUCH_SPEED_MOD`, the stalking gait.
const CROUCH_SPEED_MODIFIER: f64 = 0.6;

/// Vanilla `Ocelot::WALK_SPEED_MOD`, the ordinary gait and the tempt/stroll speed.
const WALK_SPEED_MODIFIER: f64 = 0.8;

/// Vanilla `Ocelot::SPRINT_SPEED_MOD`, the charge and the flee gait.
const SPRINT_SPEED_MODIFIER: f64 = 1.33;

/// Distance a player must be within to feed an ocelot, vanilla's bare `9.0` squared distance.
const FEEDING_DISTANCE_SQR: f64 = 9.0;

/// Chance denominator for earning trust: vanilla's `random.nextInt(3) == 0`.
const TRUST_CHANCE_DENOMINATOR: i32 = 3;

/// How far an untrusting ocelot flees a player from, vanilla's `16.0F`.
const PLAYER_AVOID_DISTANCE: f32 = 16.0;

/// Vanilla `LeapAtTargetGoal(this, 0.3F)`.
const LEAP_VERTICAL_VELOCITY: f32 = 0.3;

/// Vanilla `LookAtPlayerGoal(this, Player.class, 10.0F)`.
const LOOK_AT_PLAYER_RANGE: f32 = 10.0;

/// Vanilla `WaterAvoidingRandomStrollGoal(this, 0.8, 1.0000001E-5F)`.
///
/// The probability is written as a float one ULP above zero rather than as zero, which makes the
/// stroll effectively never pick a dry-land target while still not being literally impossible.
const STROLL_PROBABILITY: f32 = 1.000_000_1E-5;

/// Ticks an untrusting ocelot survives before it may despawn, vanilla's `tickCount > 2400`.
const DESPAWN_GRACE_TICKS: i32 = 2400;

/// Vanilla `Ocelot.getAmbientSoundInterval`.
const AMBIENT_SOUND_INTERVAL: i32 = 900;

const OCELOT_BABY_PASSENGER_ATTACHMENTS: [EntityAttachmentPoint; 1] =
    [EntityAttachmentPoint::new(0.0, 0.312_5, 0.0)];
const OCELOT_BABY_WIDTH: f32 = 0.3;
const OCELOT_BABY_HEIGHT: f32 = 0.35;
const OCELOT_BABY_EYE_HEIGHT: f32 = 0.343_75;

const OCELOT_BABY_DIMENSIONS: EntityDimensions = EntityDimensions::new_with_attachments(
    OCELOT_BABY_WIDTH,
    OCELOT_BABY_HEIGHT,
    OCELOT_BABY_EYE_HEIGHT,
    EntityAttachments::new(&OCELOT_BABY_PASSENGER_ATTACHMENTS, &[], &[], &[]),
);
const DEFAULT_STEP_HEIGHT: f32 = 0.6;

#[entity_behavior(class = "Ocelot")]
/// Vanilla ocelot entity with its synced trust flag.
pub struct OcelotEntity {
    base: EntityBase,
    entity_type: EntityTypeRef,
    living_base: LivingEntityBase,
    mob_base: MobBase,
    ageable_base: AgeableMobBase,
    animal_base: AnimalBase,
    entity_data: SyncMutex<OcelotEntityData>,
}

// SAFETY: This key is owned by Steel and uniquely identifies `OcelotEntity`.
unsafe impl DowncastType for OcelotEntity {
    const TYPE_KEY: DowncastTypeKey = DowncastTypeKey::new("steel:entity/ocelot");
}

impl OcelotEntity {
    /// Creates a new ocelot at runtime.
    #[must_use]
    pub fn new(entity_type: EntityTypeRef, id: i32, position: DVec3, world: Weak<World>) -> Self {
        Self::new_with_base(
            EntityBase::new(id, position, entity_type.dimensions, world),
            entity_type,
        )
    }

    /// Reconstructs an ocelot from persisted base entity state.
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
        let animal_base = AnimalBase::new();
        AnimalBase::initialize_pathfinding_malus(&mob_base);
        let mut entity_data = OcelotEntityData::new();
        living_base.initialize_synced_data(&mut entity_data);

        {
            // Vanilla's priorities, with its gaps preserved: 2, 5 and 6 are deliberately empty.
            let mut goal_selector = mob_base.goal_selector().lock();
            goal_selector.add_goal(1, FloatGoal::new(&mob_base));
            goal_selector.add_goal(
                3,
                TemptGoal::new(
                    CROUCH_SPEED_MODIFIER,
                    |item_stack| {
                        REGISTRY
                            .items
                            .is_in_tag(item_stack.item(), &ItemTag::OCELOT_FOOD)
                    },
                    true,
                ),
            );
            // Vanilla adds and removes this goal from `reassessTrustingGoals` as the flag changes.
            // Its `canUse` and `canContinueToUse` both already test `!isTrusting()`, so adding it
            // once and letting the selector skip it is the same behavior without a mutable goal
            // list — and it is the only reading that survives a trust flag restored from NBT
            // before the goal list is built.
            goal_selector.add_goal(
                4,
                AvoidEntityGoal::with_selector(
                    PLAYER_AVOID_DISTANCE,
                    WALK_SPEED_MODIFIER,
                    SPRINT_SPEED_MODIFIER,
                    |living, _| {
                        living.as_player().is_some_and(|player| {
                            !living.is_spectator() && !player.has_infinite_materials()
                        })
                    },
                ),
            );
            goal_selector.add_goal(7, LeapAtTargetGoal::new(LEAP_VERTICAL_VELOCITY));
            goal_selector.add_goal(8, OcelotAttackGoal::new());
            goal_selector.add_goal(9, BreedGoal::new(WALK_SPEED_MODIFIER));
            goal_selector.add_goal(
                10,
                WaterAvoidingRandomStrollGoal::with_probability(
                    WALK_SPEED_MODIFIER,
                    STROLL_PROBABILITY,
                ),
            );
            goal_selector.add_goal(11, LookAtPlayerGoal::new(f64::from(LOOK_AT_PLAYER_RANGE)));
        }

        Self {
            base,
            entity_type,
            living_base,
            mob_base,
            ageable_base,
            animal_base,
            entity_data: SyncMutex::new(entity_data),
        }
    }

    /// Returns whether this ocelot has been fed enough to stop fleeing players.
    #[must_use]
    pub fn is_trusting(&self) -> bool {
        *self.entity_data.lock().trusting.get()
    }

    /// Sets the trust flag.
    pub fn set_trusting(&self, trusting: bool) {
        self.entity_data.lock().trusting.set(trusting);
    }

    /// Returns whether an item stack matches the vanilla ocelot food tag.
    #[must_use]
    pub fn is_food(item_stack: &ItemStack) -> bool {
        REGISTRY
            .items
            .is_in_tag(item_stack.item(), &ItemTag::OCELOT_FOOD)
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

    /// Runs vanilla's trust attempt, returning whether the interaction consumed the item.
    fn try_earn_trust(&self, player: &Player, hand: InteractionHand) -> bool {
        if self.is_trusting() {
            return false;
        }

        let is_food = {
            let inventory = player.inventory.lock();
            let held = inventory.get_item_in_hand(hand);
            Self::is_food(held)
        };
        if !is_food {
            return false;
        }

        if player.position().distance_squared(self.position()) >= FEEDING_DISTANCE_SQR {
            return false;
        }

        self.use_player_item(player, hand);

        if rand::random_range(0..TRUST_CHANCE_DENOMINATOR) == 0 {
            self.set_trusting(true);
            self.broadcast_entity_event(EntityStatus::TrustingSucceeded);
        } else {
            self.broadcast_entity_event(EntityStatus::TrustingFailed);
        }

        true
    }
}

impl Entity for OcelotEntity {
    fn base(&self) -> &EntityBase {
        &self.base
    }

    fn entity_type(&self) -> EntityTypeRef {
        self.entity_type
    }

    fn base_tick(&self) {
        Mob::base_tick_mob(self);
    }

    fn dimensions_for_pose(&self, _pose: EntityPose) -> EntityDimensions {
        let scale = LivingEntity::get_scale(self);
        if AgeableMob::is_baby(self) {
            OCELOT_BABY_DIMENSIONS.scale(scale)
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

    fn max_up_step(&self) -> f32 {
        self.attributes()
            .lock()
            .get_value(vanilla_attributes::STEP_HEIGHT)
            .unwrap_or(f64::from(DEFAULT_STEP_HEIGHT)) as f32
    }

    fn sound_source(&self) -> SoundSource {
        SoundSource::Neutral
    }

    /// Vanilla `Ocelot.isSteppingCarefully`, which adds crouching to the base test.
    fn is_stepping_carefully(&self) -> bool {
        self.is_crouching() || self.is_suppressing_bounce()
    }

    fn play_step_sound(&self, _pos: BlockPos, _block_state: BlockStateId) {}

    fn save_additional(&self, nbt: &mut NbtCompound) {
        self.save_mob(nbt);
        self.save_ageable_mob(nbt);
        self.save_animal(nbt);
        nbt.insert("Trusting", self.is_trusting());
    }

    fn load_additional(&self, nbt: BorrowedNbtCompoundView<'_, '_>) {
        self.load_mob(nbt);
        self.load_ageable_mob(nbt);
        self.load_animal(nbt);
        self.set_trusting(nbt.byte("Trusting").is_some_and(|value| value != 0));
    }
}

impl LivingEntity for OcelotEntity {
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

    fn hurt_sound(&self, _source: &DamageSource) -> Option<SoundEventRef> {
        Some(&sound_events::ENTITY_OCELOT_HURT)
    }

    fn death_sound(&self) -> Option<SoundEventRef> {
        Some(&sound_events::ENTITY_OCELOT_DEATH)
    }

    fn server_ai_step(&self) {
        Mob::mob_server_ai_step(self);
    }

    fn ai_step(&self) -> Option<MoveResult> {
        let result = self.default_ai_step();

        AgeableMob::tick_ageable_mob(self);
        Animal::tick_animal_love(self);
        result
    }
}

impl AgeableMob for OcelotEntity {
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

impl Animal for OcelotEntity {
    fn animal_base(&self) -> &AnimalBase {
        &self.animal_base
    }

    fn is_food(&self, item_stack: &ItemStack) -> bool {
        OcelotEntity::is_food(item_stack)
    }
}

impl Mob for OcelotEntity {
    fn mob_base(&self) -> &MobBase {
        &self.mob_base
    }

    fn tick_goal_selectors(&self) {
        PathfinderMob::tick_pathfinder_goal_selectors(self);
    }

    fn tick_path_navigation(&self) {
        PathfinderMob::tick_pathfinder_path_navigation(self);
    }

    /// Vanilla `Ocelot.customServerAiStep`, the pose-from-gait mapping.
    ///
    /// Vanilla compares the move control's speed against its three constants with `==` on doubles.
    /// That is exact here and in vanilla for the same reason: every writer of the field passes one
    /// of these three literals, so the bit patterns match rather than merely being close.
    fn custom_server_ai_step(&self) {
        let (has_wanted, speed_modifier) = {
            let controls = self.mob_base.controls().lock();
            (
                controls.move_control.operation() == MoveControlOperation::MoveTo,
                controls.move_control.speed_modifier(),
            )
        };

        let (pose, sprinting) = if !has_wanted {
            (EntityPose::Standing, false)
        } else if speed_modifier == CROUCH_SPEED_MODIFIER {
            (EntityPose::Sneaking, false)
        } else if speed_modifier == SPRINT_SPEED_MODIFIER {
            (EntityPose::Standing, true)
        } else {
            (EntityPose::Standing, false)
        };

        self.set_pose(pose);
        LivingEntity::set_sprinting(self, sprinting);

        Animal::custom_server_ai_step_animal(self);
    }

    fn ambient_sound(&self) -> Option<SoundEventRef> {
        Some(&sound_events::ENTITY_OCELOT_AMBIENT)
    }

    fn ambient_sound_interval(&self) -> i32 {
        AMBIENT_SOUND_INTERVAL
    }

    /// Vanilla `Ocelot.removeWhenFarAway`.
    ///
    /// An ocelot a player has befriended is permanent; one that is still wild despawns, but only
    /// after two minutes, which is what gives a player time to find and feed it.
    fn remove_when_far_away(&self, _dist_sqr: f64) -> bool {
        !self.is_trusting() && self.tick_count() > DESPAWN_GRACE_TICKS
    }

    fn finalize_spawn(
        &self,
        world: &Arc<World>,
        spawn_reason: EntitySpawnReason,
        group_data: Option<SpawnGroupData>,
    ) -> Option<SpawnGroupData> {
        // Vanilla substitutes a group data with a baby chance of 1.0 when the caller brought none,
        // which is why wild ocelots arrive as one adult leading a litter: the first of the group
        // sees group_size 0 and stays adult, and every one after it is certain to be a kitten.
        let group_data = group_data.or(Some(SpawnGroupData::AgeableMob(
            AgeableMobGroupData::with_baby_spawn_chance(1.0),
        )));

        self.finalize_spawn_ageable_mob(world, spawn_reason, group_data)
    }

    fn mob_interact(&self, player: &Player, hand: InteractionHand) -> InteractionResult {
        if self.try_earn_trust(player, hand) {
            return InteractionResult::Success;
        }

        Animal::mob_interact_animal(self, player, hand)
    }

    fn mob_flags(&self) -> i8 {
        *self.entity_data.lock().mob().mob_flags.get()
    }

    fn set_mob_flags(&self, flags: i8) {
        self.entity_data.lock().mob_mut().mob_flags.set(flags);
    }
}

impl PathfinderMob for OcelotEntity {}

#[cfg(test)]
mod tests;
