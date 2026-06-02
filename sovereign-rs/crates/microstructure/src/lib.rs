//! # sovereign-microstructure
//!
//! The "Gray-Area physics" layer of Sovereign V331:
//!
//! * [`clock`] — **Information-Driven Clocks**: volume bars and tick-imbalance
//!   bars that replace chronological OHLCV, so time dilates under stress.
//! * [`turbulence`] — **Navier-Stokes microstructure**: Hurst exponent and a
//!   microstructure Reynolds number to detect the laminar→turbulent transition
//!   that should flip the engine from directional trading to market-making.
//!
//! Pure, dependency-light, and exhaustively property-tested.
#![forbid(unsafe_code)]

pub mod clock;
pub mod turbulence;

pub use clock::{clock_intensity, tick_imbalance_bars, volume_bars, InfoBar, Tick};
pub use turbulence::{classify, hurst_exponent, microstructure_reynolds, FlowRegime, Turbulence};
