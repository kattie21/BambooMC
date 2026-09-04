//! Passive entity implementations.
/// Those mobs are passive creatures that run away when attacked by a player.
mod chicken;
mod cow;
mod mooshroom;
mod ocelot;
mod pig;
mod sheep;
mod squid;

pub use chicken::ChickenEntity;
pub use cow::CowEntity;
pub use mooshroom::{MooshroomEntity, MooshroomVariant};
pub use ocelot::OcelotEntity;
pub use pig::PigEntity;
pub use sheep::SheepEntity;
pub use squid::SquidEntity;
