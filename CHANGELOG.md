# Changelog

All notable changes to this project will be documented here.

## [1.1.0] - 2025-11-09
### Added
- Added `clean_signal_columns()` function to sanitize signal-related columns before saving to SQLite.
- Integrated NaN-safe handling in `generate_ema_signals()` pipeline.
- Ensured text/date fields are blank instead of `NaN` or `None`.

### Fixed
- Prevented numeric columns (Price, Vol, Ratios) from being overwritten with zeros during cleanup.

---

## [1.0.0] - 2025-10-01
### Initial Release
- Added `generate_ema_signals()` pipeline for EMA crossovers.
- Added metadata and accumulation/distribution modules.
