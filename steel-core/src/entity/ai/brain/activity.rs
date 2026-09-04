//! Vanilla's `Activity`, the mode a brain is currently in.
//!
//! Ports `net.minecraft.world.entity.schedule.Activity`. Vanilla registers these into
//! `BuiltInRegistries.ACTIVITY`, but the registry carries no data beyond the name — every
//! consumer compares identity — so Steel models it as an enum rather than generating a registry
//! with empty rows.
//!
//! A brain holds a set of *core* activities that always run plus at most one non-core activity,
//! and a behavior is only started when one of its activities is active. That is the whole
//! mechanism by which a villager stops strolling the moment it panics.

/// One mode of a brain, vanilla's `Activity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Activity {
    /// Always active; holds the behaviors that must run in every mode.
    Core,
    /// The default non-core activity, and the fallback when no other one's requirements hold.
    Idle,
    /// Working at a job site.
    Work,
    /// Baby villagers playing.
    Play,
    /// Sleeping.
    Rest,
    /// Gathering at the village bell.
    Meet,
    /// Fleeing a hostile.
    Panic,
    /// Taking part in a raid.
    Raid,
    /// Reacting to a raid that has not started.
    PreRaid,
    /// Hiding indoors.
    Hide,
    /// Fighting a target.
    Fight,
    /// Celebrating a survived raid.
    Celebrate,
    /// A piglin admiring a held item.
    AdmireItem,
    /// Avoiding an entity.
    Avoid,
    /// Riding another entity.
    Ride,
    /// An axolotl playing dead.
    PlayDead,
    /// A goat or frog preparing a long jump.
    LongJump,
    /// A goat ramming.
    Ram,
    /// A frog's tongue attack.
    Tongue,
    /// Swimming.
    Swim,
    /// Laying spawn.
    LaySpawn,
    /// A sniffer sniffing.
    Sniff,
    /// A warden investigating a disturbance.
    Investigate,
    /// A warden roaring.
    Roar,
    /// A warden emerging from the ground.
    Emerge,
    /// A warden digging back down.
    Dig,
}

impl Activity {
    /// Returns the vanilla registry name, which is also what the debug overlay shows.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::Idle => "idle",
            Self::Work => "work",
            Self::Play => "play",
            Self::Rest => "rest",
            Self::Meet => "meet",
            Self::Panic => "panic",
            Self::Raid => "raid",
            Self::PreRaid => "pre_raid",
            Self::Hide => "hide",
            Self::Fight => "fight",
            Self::Celebrate => "celebrate",
            Self::AdmireItem => "admire_item",
            Self::Avoid => "avoid",
            Self::Ride => "ride",
            Self::PlayDead => "play_dead",
            Self::LongJump => "long_jump",
            Self::Ram => "ram",
            Self::Tongue => "tongue",
            Self::Swim => "swim",
            Self::LaySpawn => "lay_spawn",
            Self::Sniff => "sniff",
            Self::Investigate => "investigate",
            Self::Roar => "roar",
            Self::Emerge => "emerge",
            Self::Dig => "dig",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activity_names_match_vanilla_registry_keys() {
        assert_eq!(Activity::Core.name(), "core");
        assert_eq!(Activity::Idle.name(), "idle");
        assert_eq!(Activity::PreRaid.name(), "pre_raid");
        assert_eq!(Activity::AdmireItem.name(), "admire_item");
        assert_eq!(Activity::PlayDead.name(), "play_dead");
        assert_eq!(Activity::LongJump.name(), "long_jump");
        assert_eq!(Activity::LaySpawn.name(), "lay_spawn");
    }

    #[test]
    fn core_and_idle_are_distinct_and_hashable() {
        let mut set = rustc_hash::FxHashSet::default();
        assert!(set.insert(Activity::Core));
        assert!(set.insert(Activity::Idle));
        assert!(!set.insert(Activity::Core));
        assert_eq!(set.len(), 2);
    }
}
