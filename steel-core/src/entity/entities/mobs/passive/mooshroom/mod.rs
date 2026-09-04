//! Vanilla `MushroomCow` — the mooshroom, an [`AbstractCow`] that carries a mushroom variant.
//!
//! Ports `net.minecraft.world.entity.animal.cow.MushroomCow`. Vanilla factors the shared cow
//! goals, food tag and milking onto `AbstractCow`; Steel has no `AbstractCow` trait, so this
//! repeats that list exactly as [`CowEntity`](super::cow::CowEntity) does rather than inventing a
//! base class for two implementors. The two files must stay in step: a change to `AbstractCow.java`
//! belongs in both.
//!
//! Vanilla's sound handling differs from the cow's in a way worth naming. `AbstractCow`'s default
//! `getSoundVariant` is `COW_SOUNDS.get(CLASSIC)`, and only `Cow` overrides it with its own synced
//! variant. A mooshroom therefore always uses the classic cow sounds, which is why
//! `MushroomCowEntityData` has no `sound_variant` field and this file reads the flat
//! `ENTITY_COW_*` events instead of a registry lookup.

use std::sync::{Arc, Weak};

use glam::DVec3;
use simdnbt::borrow::NbtCompound as BorrowedNbtCompoundView;
use simdnbt::owned::NbtCompound;
use steel_macros::entity_behavior;
use steel_protocol::packets::game::SoundSource;
use steel_registry::entity_type::{
    EntityAttachmentPoint, EntityAttachments, EntityDimensions, EntityTypeRef,
};
use steel_registry::item_stack::ItemStack;
use steel_registry::loot_table::LootTableRef;
use steel_registry::sound_event::SoundEventRef;
use steel_registry::vanilla_entity_data::MushroomCowEntityData;
use steel_registry::vanilla_item_tags::ItemTag;
use steel_registry::{
    REGISTRY, TaggedRegistryExt, sound_events, vanilla_attributes, vanilla_items,
    vanilla_loot_tables,
};
use steel_utils::locks::SyncMutex;
use steel_utils::types::InteractionHand;
use steel_utils::{
    BlockPos, BlockStateId, Downcast as _, DowncastType, DowncastTypeKey, Identifier,
};

use crate::behavior::InteractionResult;
use crate::entity::ai::goal::{
    BreedGoal, FloatGoal, FollowParentGoal, LookAtPlayerGoal, PanicGoal, RandomLookAroundGoal,
    TemptGoal, WaterAvoidingRandomStrollGoal,
};
use crate::entity::damage::DamageSource;
use crate::entity::living_entity::shearing_loot_items_with_rng;
use crate::entity::{
    AgeableMob, AgeableMobBase, Animal, AnimalBase, Entity, EntityBase, EntityBaseLoad, EntityPose,
    EntitySpawnReason, EntitySyncedData, LivingEntity, LivingEntityBase, Mob, MobBase,
    PathfinderMob, SpawnGroupData,
};
use crate::physics::MoveResult;
use crate::player::Player;
use crate::world::World;

/// Vanilla `MushroomCow.BABY_DIMENSIONS`, the same box `AbstractCow` gives a calf.
const MOOSHROOM_BABY_PASSENGER_ATTACHMENTS: [EntityAttachmentPoint; 1] =
    [EntityAttachmentPoint::new(0.0, 0.75, 0.0)];
const MOOSHROOM_BABY_WIDTH: f32 = 0.45;
const MOOSHROOM_BABY_HEIGHT: f32 = 0.7;
const MOOSHROOM_BABY_EYE_HEIGHT: f32 = 0.69;

const MOOSHROOM_BABY_DIMENSIONS: EntityDimensions = EntityDimensions::new_with_attachments(
    MOOSHROOM_BABY_WIDTH,
    MOOSHROOM_BABY_HEIGHT,
    MOOSHROOM_BABY_EYE_HEIGHT,
    EntityAttachments::new(&MOOSHROOM_BABY_PASSENGER_ATTACHMENTS, &[], &[], &[]),
);

const DEFAULT_STEP_HEIGHT: f32 = 0.6;

/// Vanilla `MushroomCow.MUTATE_CHANCE`: two same-colored parents have a 1-in-1024 chance of
/// producing the opposite color, which is the only way to get a brown mooshroom by breeding.
const MUTATE_CHANCE: i32 = 1024;

/// Vanilla `MushroomCow.Variant`, whose ids are the wire values of the synced `variant_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MooshroomVariant {
    /// Vanilla `Variant.RED`, and `Variant.DEFAULT`.
    Red,
    /// Vanilla `Variant.BROWN`, the variant that accepts a flower for suspicious stew.
    Brown,
}

impl MooshroomVariant {
    /// Vanilla `Variant.DEFAULT`.
    pub const DEFAULT: Self = Self::Red;

    /// Vanilla `Variant.id`.
    #[must_use]
    pub const fn id(self) -> i32 {
        match self {
            Self::Red => 0,
            Self::Brown => 1,
        }
    }

    /// Vanilla `Variant.byId`, whose `ByIdMap` is built with `OutOfBoundsStrategy.CLAMP`.
    ///
    /// Clamping rather than wrapping is why a corrupt or future id reads back as an end of the
    /// range instead of an arbitrary variant.
    #[must_use]
    pub const fn by_id(id: i32) -> Self {
        if id <= 0 { Self::Red } else { Self::Brown }
    }

    /// Vanilla `Variant.getSerializedName`, the `Type` NBT string.
    #[must_use]
    pub const fn serialized_name(self) -> &'static str {
        match self {
            Self::Red => "red",
            Self::Brown => "brown",
        }
    }

    /// Parses vanilla's `Variant.CODEC`, which rejects anything outside the two names.
    #[must_use]
    pub fn from_serialized_name(name: &str) -> Option<Self> {
        match name {
            "red" => Some(Self::Red),
            "brown" => Some(Self::Brown),
            _ => None,
        }
    }

    /// The shearing loot table vanilla resolves for this variant.
    ///
    /// Vanilla passes `BuiltInLootTables.SHEAR_MOOSHROOM` and lets the table's own variant
    /// predicate pick the mushroom; Steel's generated tables are already split per variant, so the
    /// selection happens here instead of inside the table.
    #[must_use]
    pub fn shearing_loot_table(self) -> LootTableRef {
        match self {
            Self::Red => &vanilla_loot_tables::SHEARING_MOOSHROOM_RED,
            Self::Brown => &vanilla_loot_tables::SHEARING_MOOSHROOM_BROWN,
        }
    }

    /// The opposite variant, for vanilla's lightning conversion and breeding mutation.
    #[must_use]
    pub const fn flipped(self) -> Self {
        match self {
            Self::Red => Self::Brown,
            Self::Brown => Self::Red,
        }
    }
}

#[entity_behavior(class = "MushroomCow")]
/// Vanilla mooshroom entity with its synced mushroom variant.
pub struct MooshroomEntity {
    base: EntityBase,
    entity_type: EntityTypeRef,
    living_base: LivingEntityBase,
    mob_base: MobBase,
    ageable_base: AgeableMobBase,
    animal_base: AnimalBase,
    entity_data: SyncMutex<MushroomCowEntityData>,
}

// SAFETY: This key is owned by Steel and uniquely identifies `MooshroomEntity`.
unsafe impl DowncastType for MooshroomEntity {
    const TYPE_KEY: DowncastTypeKey = DowncastTypeKey::new("steel:entity/mooshroom");
}

impl MooshroomEntity {
    /// Creates a new mooshroom at runtime.
    #[must_use]
    pub fn new(entity_type: EntityTypeRef, id: i32, position: DVec3, world: Weak<World>) -> Self {
        Self::new_with_base(
            EntityBase::new(id, position, entity_type.dimensions, world),
            entity_type,
        )
    }

    /// Reconstructs a mooshroom from persisted base entity state.
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
        let mut entity_data = MushroomCowEntityData::new();
        living_base.initialize_synced_data(&mut entity_data);

        {
            // Vanilla `AbstractCow.registerGoals`, priorities and speeds unchanged.
            let mut goal_selector = mob_base.goal_selector().lock();
            goal_selector.add_goal(0, FloatGoal::new(&mob_base));
            goal_selector.add_goal(1, PanicGoal::new(2.0));
            goal_selector.add_goal(2, BreedGoal::new(1.0));
            goal_selector.add_goal(
                3,
                TemptGoal::new(
                    1.25,
                    |item_stack| {
                        REGISTRY
                            .items
                            .is_in_tag(item_stack.item(), &ItemTag::COW_FOOD)
                    },
                    false,
                ),
            );
            goal_selector.add_goal(4, FollowParentGoal::new(1.25));
            goal_selector.add_goal(5, WaterAvoidingRandomStrollGoal::new(1.0));
            goal_selector.add_goal(6, LookAtPlayerGoal::new(6.0));
            goal_selector.add_goal(7, RandomLookAroundGoal::new());
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

    /// Returns vanilla `MushroomCow.getVariant`.
    #[must_use]
    pub fn variant(&self) -> MooshroomVariant {
        MooshroomVariant::by_id(*self.entity_data.lock().mushroom_cow().variant_type.get())
    }

    /// Sets vanilla `MushroomCow.setVariant`.
    pub fn set_variant(&self, variant: MooshroomVariant) {
        self.entity_data
            .lock()
            .mushroom_cow_mut()
            .variant_type
            .set(variant.id());
    }

    /// Returns whether an item stack matches the vanilla cow food tag.
    ///
    /// Mooshrooms share `AbstractCow.isFood`, so wheat feeds both them and cows.
    #[must_use]
    pub fn is_food(item_stack: &ItemStack) -> bool {
        REGISTRY
            .items
            .is_in_tag(item_stack.item(), &ItemTag::COW_FOOD)
    }

    /// Returns vanilla `MushroomCow.readyForShearing`.
    #[must_use]
    pub fn ready_for_shearing(&self) -> bool {
        !AgeableMob::is_baby(self)
    }

    /// Runs the drop half of vanilla `MushroomCow.shear`.
    ///
    /// Vanilla's `shear` plays the sound, converts the mooshroom to a cow, and drops the mushrooms
    /// from inside the conversion callback. Steel has no `Entity.convertTo` yet, so this performs
    /// the sound and the drops and reports whether the caller still owes the conversion. Splitting
    /// it this way keeps the drop behavior correct and testable now, and leaves exactly one seam
    /// for `convertTo` to land in rather than scattering the work.
    pub fn shear_drops(&self, world: &World, tool: &ItemStack) {
        world.play_sound_at(
            &sound_events::ENTITY_MOOSHROOM_SHEAR,
            SoundSource::Players,
            self.position(),
            1.0,
            1.0,
            None,
        );

        let mut rng = rand::rng();
        let drops = shearing_loot_items_with_rng(
            self,
            self.variant().shearing_loot_table(),
            tool,
            &mut rng,
        );
        for drop in drops {
            self.spawn_shearing_drop(&drop);
        }
    }

    /// Drops one stack from shearing one item at a time, with vanilla's jitter.
    ///
    /// Vanilla's lambda spawns a `count(1)` item entity per count unit rather than one stack, so a
    /// sheared mooshroom scatters its mushrooms instead of dropping them as a pile.
    fn spawn_shearing_drop(&self, drop: &ItemStack) {
        for _ in 0..drop.count() {
            let Some(item_entity) = self.spawn_at_location(drop.copy_with_count(1), 1.0) else {
                continue;
            };
            let jitter = DVec3::new(
                (rand::random::<f64>() - rand::random::<f64>()) * 0.1,
                rand::random::<f64>() * 0.05,
                (rand::random::<f64>() - rand::random::<f64>()) * 0.1,
            );
            item_entity.set_velocity(item_entity.velocity() + jitter);
        }
    }

    /// Vanilla `MushroomCow.getOffspringVariant`.
    ///
    /// Two same-colored parents mutate to the opposite color once in [`MUTATE_CHANCE`]; otherwise
    /// the calf takes one parent's color at random. Note the mutation test requires the parents to
    /// *match*, so a red and a brown never mutate — they simply pick a side.
    #[must_use]
    pub fn offspring_variant(&self, mate: &Self) -> MooshroomVariant {
        Self::offspring_variant_from(self.variant(), mate.variant())
    }

    /// [`Self::offspring_variant`] over two already-read variants.
    ///
    /// Breeding reads both parents through a `&dyn Animal`, where the mate's variant has to be
    /// recovered by downcast before the draw; taking the values keeps that read at the call site.
    #[must_use]
    fn offspring_variant_from(
        variant: MooshroomVariant,
        mate_variant: MooshroomVariant,
    ) -> MooshroomVariant {
        if variant == mate_variant && rand::random_range(0..MUTATE_CHANCE) == 0 {
            variant.flipped()
        } else if rand::random::<bool>() {
            variant
        } else {
            mate_variant
        }
    }

    /// Handles the bowl branch of vanilla `MushroomCow.mobInteract`.
    ///
    /// Steel has no `SuspiciousStewEffects` data component yet, so only the plain mushroom-stew
    /// path is implemented; a mooshroom that has been fed a flower is not yet representable, so
    /// there is no state under which the suspicious branch could fire.
    fn try_milk_stew(&self, player: &Player, hand: InteractionHand) -> bool {
        if AgeableMob::is_baby(self) {
            return false;
        }

        let is_bowl = {
            let inventory = player.inventory.lock();
            inventory.get_item_in_hand(hand).is(&vanilla_items::BOWL)
        };
        if !is_bowl {
            return false;
        }

        let overflow = {
            let mut inventory = player.inventory.lock();
            inventory.apply_filled_result(
                hand,
                ItemStack::new(&vanilla_items::MUSHROOM_STEW),
                player.has_infinite_materials(),
                false,
            )
        };

        self.play_sound(&sound_events::ENTITY_MOOSHROOM_MILK, 1.0, 1.0);

        if !overflow.is_empty() {
            let _ = player.drop_item(overflow, false, false);
        }

        true
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
}

impl Entity for MooshroomEntity {
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
            MOOSHROOM_BABY_DIMENSIONS.scale(scale)
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

    fn play_step_sound(&self, _pos: BlockPos, _block_state: BlockStateId) {
        self.play_sound(&sound_events::ENTITY_COW_STEP, 0.15, 1.0);
    }

    fn save_additional(&self, nbt: &mut NbtCompound) {
        self.save_mob(nbt);
        self.save_ageable_mob(nbt);
        self.save_animal(nbt);
        nbt.insert("Type", self.variant().serialized_name());
    }

    fn load_additional(&self, nbt: BorrowedNbtCompoundView<'_, '_>) {
        self.load_mob(nbt);
        self.load_ageable_mob(nbt);
        self.load_animal(nbt);

        // Vanilla's codec read falls back to DEFAULT for a missing or unrecognized name.
        let variant = nbt
            .string("Type")
            .and_then(|name| MooshroomVariant::from_serialized_name(name.to_str().as_ref()))
            .unwrap_or(MooshroomVariant::DEFAULT);
        self.set_variant(variant);
    }
}

impl LivingEntity for MooshroomEntity {
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
        Some(&sound_events::ENTITY_COW_HURT)
    }

    fn death_sound(&self) -> Option<SoundEventRef> {
        Some(&sound_events::ENTITY_COW_DEATH)
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

impl AgeableMob for MooshroomEntity {
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

impl Animal for MooshroomEntity {
    fn animal_base(&self) -> &AnimalBase {
        &self.animal_base
    }

    fn is_food(&self, item_stack: &ItemStack) -> bool {
        MooshroomEntity::is_food(item_stack)
    }

    fn breed_variant_key(&self) -> Option<&Identifier> {
        None
    }

    fn set_breed_variant_key(&self, _key: &Identifier) -> bool {
        false
    }

    fn initialize_breed_offspring(&self, partner: &dyn Animal, offspring: &dyn Animal) {
        // Vanilla resolves the calf's color in `getBreedOffspring` from both parents' variants,
        // which are not registry keys, so the shared `breed_variant_key` seam cannot carry them.
        let own_variant = self.variant();
        let mate_variant = partner
            .downcast_ref::<Self>()
            .map_or(own_variant, Self::variant);
        if let Some(offspring) = offspring.downcast_ref::<Self>() {
            offspring.set_variant(Self::offspring_variant_from(own_variant, mate_variant));
        }
    }
}

impl Mob for MooshroomEntity {
    fn mob_base(&self) -> &MobBase {
        &self.mob_base
    }

    fn tick_goal_selectors(&self) {
        PathfinderMob::tick_pathfinder_goal_selectors(self);
    }

    fn tick_path_navigation(&self) {
        PathfinderMob::tick_pathfinder_path_navigation(self);
    }

    fn custom_server_ai_step(&self) {
        Animal::custom_server_ai_step_animal(self);
    }

    fn ambient_sound(&self) -> Option<SoundEventRef> {
        Some(&sound_events::ENTITY_COW_AMBIENT)
    }

    fn finalize_spawn(
        &self,
        world: &Arc<World>,
        spawn_reason: EntitySpawnReason,
        group_data: Option<SpawnGroupData>,
    ) -> Option<SpawnGroupData> {
        self.finalize_spawn_ageable_mob(world, spawn_reason, group_data)
    }

    fn mob_interact(&self, player: &Player, hand: InteractionHand) -> InteractionResult {
        if self.try_milk_stew(player, hand) {
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

impl PathfinderMob for MooshroomEntity {}

#[cfg(test)]
mod tests;
