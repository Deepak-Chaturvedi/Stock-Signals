# Changelog

All notable changes to this project will be documented here.

## [1.2.1] - 2026-04-15
### Fixed
- Adjust Stock Signal Prices for corporate actions - split/bonus etc. Reload full SIGNAL_RETURNS file everyday instead of inserting new rows. 
-  Rename COMPANY_DETAILS table to STOCK_DETAILS 
- Fixed pandas to use 2.2.2 version 

### Added

- Add a table STOCK_PRICES which contains stock prices for all signals for current date and updates daily
- Use 'Updated Signal Price' to calculate return for 1week/2week and so on


## [1.1.1] - 2025-11-10
### Fixed
- Refreshed stock database and notebook outputs to reflect analysis on a larger set of tickers and a shorter analysis period (15 days).
- Updated signal generation,  and error handling for failed downloads. 


## [1.1.0] - 2025-11-09
### Added
- Added `generate_ema_signals()` to add ema_crossovers in past 30 days into the Database
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
