#!/usr/bin/env Rscript

# NASCAR Data Update Script
# This script updates the nascaR.data package and exports updated data

# Load required libraries and install if needed
if (!require("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

if (!require("arrow", quietly = TRUE)) {
  install.packages("arrow")
}

library(remotes)
library(arrow)

# Function to log messages with timestamps
log_message <- function(message) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(paste0("[", timestamp, "] ", message, "\n"))
}

# Function to check if we're in the right directory
check_project_structure <- function() {
  if (!dir.exists("data/raw")) {
    log_message("Creating data/raw directory...")
    dir.create("data/raw", recursive = TRUE)
  }

  if (!dir.exists("data/processed")) {
    log_message("Creating data/processed directory...")
    dir.create("data/processed", recursive = TRUE)
  }
}

# Main execution
main <- function() {
  log_message("Starting NASCAR data update process...")

  # Check project structure
  check_project_structure()

  # Update nascaR.data package
  log_message("Updating nascaR.data package from GitHub...")
  tryCatch({
    remotes::install_github("kyleGrealis/nascaR.data", quiet = TRUE)
    log_message("Package update completed successfully")
  }, error = function(e) {
    log_message(paste("Error updating package:", e$message))
    quit(status = 1)
  })

  # Load the package
  log_message("Loading nascaR.data package...")
  tryCatch({
    library(nascaR.data)
    log_message("Package loaded successfully")
  }, error = function(e) {
    log_message(paste("Error loading package:", e$message))
    quit(status = 1)
  })

  # Check if cup_series data exists
  if (!exists("cup_series")) {
    log_message("Error: cup_series dataset not found in package")
    log_message("Available objects in nascaR.data:")
    log_message(paste(ls("package:nascaR.data"), collapse = ", "))
    quit(status = 1)
  }

  # Get data information
  data_info <- list(
    total_records = nrow(nascaR.data::cup_series),
    seasons = paste(min(nascaR.data::cup_series$Season, na.rm = TRUE),
                    max(nascaR.data::cup_series$Season, na.rm = TRUE),
                    sep = " - "),
    unique_drivers = length(unique(nascaR.data::cup_series$Driver)),
    last_race_date = max(nascaR.data::cup_series$Season, na.rm = TRUE)
  )

  log_message(paste("Data loaded:", data_info$total_records, "records"))
  log_message(paste("Seasons covered:", data_info$seasons))
  log_message(paste("Unique drivers:", data_info$unique_drivers))

  # Export to Parquet (preferred format - faster and smaller)
  parquet_path <- "data/raw/cup_series.parquet"
  log_message("Exporting data to Parquet format...")
  tryCatch({
    arrow::write_parquet(nascaR.data::cup_series, parquet_path)
    log_message(paste("Parquet export completed:", parquet_path))
  }, error = function(e) {
    log_message(paste("Error exporting to Parquet:", e$message))

    # Fallback to CSV
    log_message("Falling back to CSV export...")
    csv_path <- "data/raw/cup_series.csv"
    write.csv(nascaR.data::cup_series, csv_path, row.names = FALSE)
    log_message(paste("CSV export completed:", csv_path))
  })

  # Create metadata file with update information
  metadata <- list(
    update_time = as.character(Sys.time()),
    update_date = as.character(Sys.Date()),
    total_records = data_info$total_records,
    seasons_covered = data_info$seasons,
    unique_drivers = data_info$unique_drivers,
    r_version = R.version.string,
    package_source = "kyleGrealis/nascaR.data",
    export_format = if (file.exists(parquet_path)) "parquet" else "csv"
  )

  # Write metadata as JSON for Python to read easily
  metadata_json <- jsonlite::toJSON(metadata, pretty = TRUE, auto_unbox = TRUE)
  writeLines(metadata_json, "data/raw/data_metadata.json")

  # Also write simple timestamp file for quick checking
  update_info <- paste("NASCAR data updated:", Sys.time(),
                       "\nTotal records:", data_info$total_records,
                       "\nSeasons:", data_info$seasons)
  writeLines(update_info, "data/raw/last_update.txt")

  log_message("Metadata files created successfully")
  log_message("NASCAR data update process completed!")

  # Return success status
  0
}

# Execute main function if script is run directly
if (!interactive()) {
  result <- main()
  quit(status = result)
}