use std::io::{self, Cursor, Write};

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

use crate::{
    codec::VarInt,
    serial::{ReadFrom, WriteTo},
};

/// The game type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[expect(missing_docs, reason = "variant names are self-explanatory")]
pub enum GameType {
    Survival = 0,
    Creative = 1,
    Adventure = 2,
    Spectator = 3,
}

impl GameType {
    /// Returns the name of the game type.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            GameType::Survival => "survival",
            GameType::Creative => "creative",
            GameType::Adventure => "adventure",
            GameType::Spectator => "spectator",
        }
    }
}

impl ReadFrom for GameType {
    fn read(data: &mut Cursor<&[u8]>) -> io::Result<Self> {
        let value = VarInt::read(data)?.0;
        match value {
            0 => Ok(GameType::Survival),
            1 => Ok(GameType::Creative),
            2 => Ok(GameType::Adventure),
            3 => Ok(GameType::Spectator),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid GameType",
            )),
        }
    }
}

impl From<GameType> for i8 {
    fn from(value: GameType) -> Self {
        value as i8
    }
}

impl From<GameType> for i32 {
    fn from(value: GameType) -> Self {
        value as i32
    }
}

impl From<GameType> for f32 {
    fn from(value: GameType) -> Self {
        f32::from(value as i8)
    }
}

impl From<i8> for GameType {
    fn from(value: i8) -> Self {
        match value {
            1 => GameType::Creative,
            2 => GameType::Adventure,
            3 => GameType::Spectator,
            _ => GameType::Survival,
        }
    }
}

impl From<i32> for GameType {
    fn from(value: i32) -> Self {
        match value {
            1 => GameType::Creative,
            2 => GameType::Adventure,
            3 => GameType::Spectator,
            _ => GameType::Survival,
        }
    }
}

impl From<f32> for GameType {
    fn from(value: f32) -> Self {
        match value {
            1. => GameType::Creative,
            2. => GameType::Adventure,
            3. => GameType::Spectator,
            _ => GameType::Survival,
        }
    }
}

/// World difficulty level.
///
/// Controls starvation damage thresholds, mob spawning behavior,
/// and various other gameplay tweaks.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Difficulty {
    /// No hostile mobs, no starvation, health regenerates quickly.
    Peaceful = 0,
    /// Hostile mobs deal less damage, starvation stops at 10 HP.
    Easy = 1,
    /// Default difficulty, starvation stops at 1 HP.
    #[default]
    Normal = 2,
    /// Hostile mobs deal more damage, starvation can kill.
    Hard = 3,
}

#[expect(clippy::match_same_arms, reason = "cause it looks better")]
impl From<u8> for Difficulty {
    fn from(value: u8) -> Self {
        match value {
            0 => Difficulty::Peaceful,
            1 => Difficulty::Easy,
            2 => Difficulty::Normal,
            3 => Difficulty::Hard,
            _ => Difficulty::Normal,
        }
    }
}

impl From<Difficulty> for u8 {
    fn from(value: Difficulty) -> Self {
        value as u8
    }
}

impl ReadFrom for Difficulty {
    fn read(data: &mut Cursor<&[u8]>) -> io::Result<Self> {
        let value = <u8 as ReadFrom>::read(data)?;
        match value {
            0 => Ok(Difficulty::Peaceful),
            1 => Ok(Difficulty::Easy),
            2 => Ok(Difficulty::Normal),
            3 => Ok(Difficulty::Hard),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid Difficulty: {value}"),
            )),
        }
    }
}

impl WriteTo for Difficulty {
    fn write(&self, writer: &mut impl Write) -> io::Result<()> {
        (*self as u8).write(writer)
    }
}

impl Serialize for Difficulty {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u8(*self as u8)
    }
}

impl<'de> Deserialize<'de> for Difficulty {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let id = u8::deserialize(deserializer)?;
        Ok(Self::from(id))
    }
}

/// Represents the hand used for an interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionHand {
    /// The main hand.
    MainHand,
    /// The off hand.
    OffHand,
}

impl ReadFrom for InteractionHand {
    fn read(data: &mut Cursor<&[u8]>) -> io::Result<Self> {
        let id = VarInt::read(data)?.0;
        match id {
            0 => Ok(InteractionHand::MainHand),
            1 => Ok(InteractionHand::OffHand),
            _ => Err(io::Error::other("Invalid InteractionHand id")),
        }
    }
}

/// Vanilla `DifficultyInstance`: the difficulty actually in force at one position.
///
/// Vanilla's own name for the number is "effective difficulty", and it is not the `Difficulty`
/// enum. It is a float that grows with three independent things: how long the *world* has been
/// running, how long a *player* has lingered near this particular chunk, and the moon. Mobs read
/// it through [`Self::special_multiplier`] to decide armor, enchantments, and buffs, which is why
/// a zombie in a base you have camped in for days is a different animal from one in a chunk you
/// just walked into.
///
/// Immutable by construction, exactly as vanilla's `@Immutable` annotation says: the float is
/// computed once in the constructor and every accessor is a pure read.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DifficultyInstance {
    base: Difficulty,
    effective_difficulty: f32,
}

impl DifficultyInstance {
    /// Vanilla `DIFFICULTY_TIME_GLOBAL_OFFSET`, the world-age grace period.
    ///
    /// Added to the total game time, so the global term stays at zero for the first 72000 ticks
    /// (an hour of play, three-and-a-bit Minecraft days) rather than ramping from the first tick.
    const TIME_GLOBAL_OFFSET: f32 = -72_000.0;

    /// Vanilla `MAX_DIFFICULTY_TIME_GLOBAL`: the world age at which the global term saturates.
    ///
    /// 1440000 ticks is 20 hours of play, or 60 Minecraft days.
    const MAX_TIME_GLOBAL: f32 = 1_440_000.0;

    /// Vanilla `MAX_DIFFICULTY_TIME_LOCAL`: the chunk `inhabitedTime` at which the local term
    /// saturates. 3600000 ticks is 50 hours spent near one chunk.
    const MAX_TIME_LOCAL: f32 = 3_600_000.0;

    /// Builds vanilla `new DifficultyInstance(base, totalGameTime, localGameTime, moonBrightness)`.
    ///
    /// `local_game_time` is the chunk's `inhabitedTime` and `moon_brightness` is
    /// `MOON_BRIGHTNESS_PER_PHASE[phase]`, both supplied by the caller because neither is knowable
    /// from a difficulty setting alone.
    #[must_use]
    pub fn new(
        base: Difficulty,
        total_game_time: i64,
        local_game_time: i64,
        moon_brightness: f32,
    ) -> Self {
        Self {
            base,
            effective_difficulty: Self::calculate(
                base,
                total_game_time,
                local_game_time,
                moon_brightness,
            ),
        }
    }

    /// Returns vanilla `getDifficulty`, the plain setting this instance was built from.
    #[must_use]
    pub const fn difficulty(self) -> Difficulty {
        self.base
    }

    /// Returns vanilla `getEffectiveDifficulty`, the scaled float.
    #[must_use]
    pub const fn effective_difficulty(self) -> f32 {
        self.effective_difficulty
    }

    /// Returns vanilla `isHard`.
    ///
    /// Compares against `Difficulty.HARD.ordinal()`, which is `3`, so this can read `true` on a
    /// `Normal` world that has run long enough — the whole point of the scaling.
    #[must_use]
    pub fn is_hard(self) -> bool {
        self.effective_difficulty >= f32::from(Difficulty::Hard as u8)
    }

    /// Returns vanilla `isHarderThan`, a strict comparison.
    #[must_use]
    pub fn is_harder_than(self, required_difficulty: f32) -> bool {
        self.effective_difficulty > required_difficulty
    }

    /// Returns vanilla `getSpecialMultiplier`, the 0.0-1.0 knob mobs actually read.
    ///
    /// Nothing special happens below an effective difficulty of 2.0, everything is maximal above
    /// 4.0, and the band between them is linear. Mob equipment, enchantment chances and zombie
    /// reinforcement odds are all scaled by this.
    #[must_use]
    pub fn special_multiplier(self) -> f32 {
        if self.effective_difficulty < 2.0 {
            0.0
        } else if self.effective_difficulty > 4.0 {
            1.0
        } else {
            (self.effective_difficulty - 2.0) / 2.0
        }
    }

    /// Vanilla `calculateDifficulty`, term for term.
    ///
    /// Two details that look like mistakes and are not. The moon term is clamped to
    /// `global_scale`, not to its own `0.25` ceiling, so in a young world the moon contributes
    /// exactly nothing however full it is. And `Easy` halves only the *local* contribution, after
    /// the moon has already been folded into it, leaving the global term at full strength.
    fn calculate(
        base: Difficulty,
        total_game_time: i64,
        local_game_time: i64,
        moon_brightness: f32,
    ) -> f32 {
        if base == Difficulty::Peaceful {
            return 0.0;
        }

        let is_hard = base == Difficulty::Hard;
        let global_scale = (((total_game_time as f32) + Self::TIME_GLOBAL_OFFSET)
            / Self::MAX_TIME_GLOBAL)
            .clamp(0.0, 1.0)
            * 0.25;

        let mut local_scale = ((local_game_time as f32) / Self::MAX_TIME_LOCAL).clamp(0.0, 1.0)
            * if is_hard { 1.0 } else { 0.75 };
        local_scale += (moon_brightness * 0.25).clamp(0.0, global_scale);
        if base == Difficulty::Easy {
            local_scale *= 0.5;
        }

        let scale = 0.75 + global_scale + local_scale;
        f32::from(base as u8) * scale
    }
}

/// Flags that control how a block update is processed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UpdateFlags(u16);

bitflags! {
    impl UpdateFlags: u16 {
        const UPDATE_NEIGHBORS = 1;
        const UPDATE_CLIENTS = 1 << 1;
        const UPDATE_INVISIBLE = 1 << 2;
        const UPDATE_IMMEDIATE = 1 << 3;
        const UPDATE_KNOWN_SHAPE = 1 << 4;
        const UPDATE_SUPPRESS_DROPS = 1 << 5;
        const UPDATE_MOVE_BY_PISTON = 1 << 6;
        const UPDATE_SKIP_SHAPE_UPDATE_ON_WIRE = 1 << 7;
        const UPDATE_SKIP_BLOCK_ENTITY_SIDEEFFECTS = 1 << 8;
        const UPDATE_SKIP_ON_PLACE = 1 << 9;

        const UPDATE_NONE = Self::UPDATE_INVISIBLE.bits() | Self::UPDATE_SKIP_BLOCK_ENTITY_SIDEEFFECTS.bits();
        const UPDATE_ALL = Self::UPDATE_NEIGHBORS.bits() | Self::UPDATE_CLIENTS.bits();
        const UPDATE_ALL_IMMEDIATE = Self::UPDATE_ALL.bits() | Self::UPDATE_IMMEDIATE.bits();
    }
}

#[cfg(test)]
mod difficulty_instance_tests {
    use super::{Difficulty, DifficultyInstance};

    /// Vanilla's saturation points, so a test reads as its own arithmetic.
    const GLOBAL_SATURATED: i64 = 1_512_000;
    const LOCAL_SATURATED: i64 = 3_600_000;

    /// Compares against a float worked from vanilla's expression, not from Steel's output.
    fn assert_effective(instance: DifficultyInstance, expected: f32) {
        let actual = instance.effective_difficulty();
        assert!(
            (actual - expected).abs() < 1e-4,
            "effective difficulty {actual} should be {expected}"
        );
    }

    #[test]
    fn peaceful_is_flat_zero_whatever_the_clocks_say() {
        // Vanilla returns before reading any of the three terms.
        let instance =
            DifficultyInstance::new(Difficulty::Peaceful, GLOBAL_SATURATED, LOCAL_SATURATED, 1.0);
        assert_effective(instance, 0.0);
        assert_eq!(instance.special_multiplier(), 0.0);
        assert!(!instance.is_hard());
    }

    #[test]
    fn a_brand_new_normal_world_sits_at_three_quarters_scale() {
        // scale = 0.75 with both clocks inside the grace period, so 2 * 0.75.
        assert_effective(DifficultyInstance::new(Difficulty::Normal, 0, 0, 0.0), 1.5);
    }

    #[test]
    fn the_global_grace_period_is_seventy_two_thousand_ticks() {
        // totalGameTime + (-72000) is still <= 0 at exactly 72000, so the global term is zero and
        // the result matches a world at tick 0.
        assert_effective(
            DifficultyInstance::new(Difficulty::Normal, 72_000, 0, 0.0),
            1.5,
        );
        // One tick past it the clamp starts moving, but only by 1/1440000 of 0.25.
        let just_after = DifficultyInstance::new(Difficulty::Normal, 72_001, 0, 0.0);
        assert!(just_after.effective_difficulty() > 1.5);
    }

    #[test]
    fn both_clocks_saturated_on_normal_reaches_vanillas_ceiling() {
        // globalScale = 0.25, localScale = 1.0 * 0.75 = 0.75, moon term clamps into globalScale.
        // scale = 0.75 + 0.25 + 0.75 = 1.75, times Normal's id of 2.
        assert_effective(
            DifficultyInstance::new(Difficulty::Normal, GLOBAL_SATURATED, LOCAL_SATURATED, 0.0),
            3.5,
        );
    }

    #[test]
    fn hard_weights_the_local_term_more_heavily_than_normal_does() {
        // The isHard branch multiplies the local clock by 1.0 rather than 0.75:
        // scale = 0.75 + 0.25 + 1.0 = 2.0, times Hard's id of 3.
        assert_effective(
            DifficultyInstance::new(Difficulty::Hard, GLOBAL_SATURATED, LOCAL_SATURATED, 0.0),
            6.0,
        );
    }

    #[test]
    fn easy_halves_only_the_local_contribution() {
        // localScale 0.75 becomes 0.375, but globalScale stays 0.25:
        // scale = 0.75 + 0.25 + 0.375 = 1.375, times Easy's id of 1.
        assert_effective(
            DifficultyInstance::new(Difficulty::Easy, GLOBAL_SATURATED, LOCAL_SATURATED, 0.0),
            1.375,
        );
    }

    #[test]
    fn the_moon_is_clamped_to_the_global_term_so_a_young_world_ignores_it() {
        // This is the quirk worth pinning: clamp(moonBrightness * 0.25, 0.0, globalScale) with
        // globalScale at zero yields zero, so a full moon over a fresh world adds nothing.
        let young_new_moon = DifficultyInstance::new(Difficulty::Normal, 0, 0, 0.0);
        let young_full_moon = DifficultyInstance::new(Difficulty::Normal, 0, 0, 1.0);
        assert_eq!(
            young_full_moon.effective_difficulty().to_bits(),
            young_new_moon.effective_difficulty().to_bits()
        );

        // Once the global term saturates at 0.25 the moon's own 0.25 ceiling binds instead, and a
        // full moon adds the whole quarter: scale = 0.75 + 0.25 + 0.25 = 1.25, times 2.
        assert_effective(
            DifficultyInstance::new(Difficulty::Normal, GLOBAL_SATURATED, 0, 1.0),
            2.5,
        );
    }

    #[test]
    fn the_special_multiplier_is_flat_outside_the_two_to_four_band() {
        // Below 2.0 nothing special happens, however long the world has run.
        assert_eq!(
            DifficultyInstance::new(Difficulty::Normal, 0, 0, 0.0).special_multiplier(),
            0.0
        );
        // A saturated Hard world is past 4.0, so it pins at 1.0.
        assert_eq!(
            DifficultyInstance::new(Difficulty::Hard, GLOBAL_SATURATED, LOCAL_SATURATED, 0.0)
                .special_multiplier(),
            1.0
        );
        // 3.5 sits inside the band: (3.5 - 2.0) / 2.0.
        let banded =
            DifficultyInstance::new(Difficulty::Normal, GLOBAL_SATURATED, LOCAL_SATURATED, 0.0);
        assert!((banded.special_multiplier() - 0.75).abs() < 1e-4);
    }

    #[test]
    fn is_hard_compares_against_the_ordinal_not_the_setting() {
        // A saturated Normal world reads 3.5, which is >= Hard's ordinal of 3, so a Normal world
        // can be "hard" in vanilla's sense. This is the assertion that stops someone
        // "simplifying" is_hard into base == Hard.
        let saturated_normal =
            DifficultyInstance::new(Difficulty::Normal, GLOBAL_SATURATED, LOCAL_SATURATED, 0.0);
        assert_eq!(saturated_normal.difficulty(), Difficulty::Normal);
        assert!(saturated_normal.is_hard());

        // A fresh Normal world reads 1.5 and is not.
        assert!(!DifficultyInstance::new(Difficulty::Normal, 0, 0, 0.0).is_hard());
    }

    #[test]
    fn is_harder_than_is_strict() {
        let instance = DifficultyInstance::new(Difficulty::Normal, 0, 0, 0.0);
        assert!(instance.is_harder_than(1.4));
        assert!(!instance.is_harder_than(1.5));
    }
}
