library(parallel)
library(arrow)
library(dplyr)
library(mirt)
source("sim_helper_functions.R")

test_name <- "swesat22b_nominal_quant"
full_data <- data.frame(arrow::read_feather(paste0("real_datasets/", test_name, ".feather")))
full_data <- full_data[complete.cases(full_data), ]
correct_cat <- scan(paste0("real_datasets/", test_name, "_correct.txt"), what = integer(), sep = "\n")

iterations <- 1:1000
n_params <- c(1000, 3000, 5000, 10000)
items_params <- c(20, 40, 80)

tasks <- expand.grid(iteration = iterations, n = n_params, items = items_params)

run_simulation <- function(params) {
  error <- NA
  i <- params[1, "iteration"]
  n <- params[1, "n"]
  items <- params[1, "items"]

  tryCatch(
    {
      # Add one to the rows/cols to adjust for python 0 indexing
      sampled_rows <- c(arrow::read_feather(paste0("simulated_datasets/sampled_rows_", test_name, "_n", n, "_items", items, ".feather"))[[i]]) + 1
      sampled_items <- arrow::read_feather(paste0("simulated_datasets/sampled_items_", test_name, "_items", items, ".feather"))$items + 1
      sampled_data <- full_data[sampled_rows, sampled_items]
      remaining_data <- full_data[-(sampled_rows), sampled_items]
    },
    error = function(e) {
      error <<- e
      log_ll <<- c(NA, NA)
    }
  )

  start_time <- Sys.time()
  tryCatch(
    {
      model <- mirt::mirt(sampled_data, model = 1, itemtype = "nominal", optimizer = "nlminb", technical = list("NCYCLES" = 700))
    },
    error = function(e) {
      error <<- paste0(error, "\n", e)
    }
  )
  execution_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  missing_categories <- sapply(seq_len(ncol(remaining_data)), function(i) {
    !(remaining_data[, i] %in% unique(sampled_data[, i]))
  })
  missing_rows <- rowSums(missing_categories) > 0
  remaining_data <- remaining_data[!missing_rows, ] # remove rows with item categories not present in the training data

  if (file.exists(paste0("ml_scores/mirt_thetas_", test_name, "_n", n, "_items", items, "_iteration", i, ".rda"))) {
    load(file = paste0("ml_scores/mirt_thetas_", test_name, "_n", n, "_items", items, "_iteration", i, ".rda"))
  } else {
    test_thetas <- mirt::fscores(model, response.pattern = remaining_data, method = "ML")
    save(test_thetas, file = paste0("ml_scores/mirt_thetas_", test_name, "_n", n, "_items", items, "_iteration", i, ".rda"))
  }

  tryCatch(
    {
      log_ll <- log_likelihood(remaining_data, model, test_thetas, theta_method = "ML", train_data = sampled_data)
    },
    error = function(e) {
      log_ll <<- c(NA, NA)
    }
  )

  tryCatch(
    {
      residuals <- calculate_residuals(model, remaining_data, test_thetas, start_0 = FALSE)
      test_residuals <- unlist(residuals, use.names = FALSE) |> mean()
    },
    error = function(e) {
      error <<- paste0(error, "\n", e)
      test_residuals <<- NA
    }
  )

  result <- data.frame(
    model = "mirt nominal",
    date = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    execution_time = execution_time,
    iteration = i - 1,
    n = n,
    items = items,
    test_log_likelihood = log_ll[1] / (nrow(remaining_data) - log_ll[2]) / items, # average over items and respondents
    test_residuals = test_residuals,
    test_obs_with_missing_cats_in_train_data = log_ll[2],
    error = error
  )

  return(result)
}

start_time_overall <- Sys.time()
cl <- makeCluster(detectCores())
clusterExport(cl, c(
  "calculate_residuals",
  "correct_cat",
  "full_data",
  "get_mirt_probs",
  "log_likelihood",
  "run_simulation",
  "test_name"
))
results <- parLapply(cl, split(tasks, seq_len(nrow(tasks))), run_simulation)
stopCluster(cl)
print(difftime(Sys.time(), start_time_overall, units = "hours"))

results_df <- do.call(rbind, results)
results_df$error <- as.character(results_df$error)

file_path <- paste0("simulation_results/r_", test_name, ".feather")
# Check if file exists and concatenate data frames
if (file.exists(file_path)) {
  existing_df <- arrow::read_feather(file_path)

  # Define columns for checking duplicates
  duplicate_columns <- c("model", "iteration", "n", "items", "error")

  # Concatenate and remove duplicates
  results_df <- rbind(existing_df, results_df)
  results_df <- results_df[!duplicated(results_df[duplicate_columns], fromLast = TRUE), ]
}

arrow::write_feather(results_df, file_path)
