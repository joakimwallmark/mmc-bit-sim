# Note: train_data is only needed when all possible scores were not available in the model training dataset
log_likelihood <- function(
    data,
    mod,
    thetas = NULL,
    inverse_coded = FALSE,
    theta_method = "ML",
    train_data = NULL) {
  no_missing <- 0
  if (is.null(train_data)) {
    train_data <- mod@Data$data
  }
  modelled_scores <- apply(train_data, 2, unique)
  missing_categories <- sapply(seq_len(ncol(data)), function(i) {
    !(data[, i] %in% unique(train_data[, i]))
  })
  missing_rows <- rowSums(missing_categories) > 0
  data <- data[!missing_rows, ]
  no_missing <- sum(missing_rows)

  if (is.null(thetas)) {
    thetas <- mirt::fscores(mod, response.pattern = data, method = theta_method)
  }
  thetas <- thetas[, grepl("^F", colnames(thetas)), drop = FALSE]
  thetas[thetas == Inf] <- 100
  thetas[thetas == -Inf] <- -100

  probs <- get_mirt_probs(thetas, mod)
  log_lik <- 0
  for (item in seq_along(probs)) {
    if ("matrix" %in% class(modelled_scores)) {
      modelled_item_scores <- sort(modelled_scores[, item])
    } else {
      modelled_item_scores <- sort(modelled_scores[[item]])
    }
    if (inverse_coded) { # if multidimensional model has reverse relationship between prob and score
      data[, item] <- max(data[, item]) - data[, item]
    }
    for (person_id in seq_len(nrow(data))) {
      if (!is.na(data[person_id, item])) {
        score_ind <- which(modelled_item_scores == data[person_id, item])
        person_log_lik <- log(probs[[item]][person_id, score_ind])
        log_lik <- log_lik + person_log_lik
      }
    }
  }
  return(c("ll" = log_lik, "test_obs_with_missing_cats_in_train_data" = no_missing))
}

get_mirt_probs <- function(theta, mod) {
  probs <- list()
  for (item in 1:mirt::extract.mirt(mod, "nitems")) {
    extr <- mirt::extract.item(mod, item)
    probs[[item]] <- mirt::probtrace(extr, theta)
  }
  probs
}

calculate_residuals <- function(model, data, thetas, start_0 = TRUE) {
  # Calculate residuals for the given model and data

  # Parameters:
  # model: A mirt model object
  # data: A matrix or dataframe representing the data
  # thetas: A numeric matrix representing thetas
  # start_0: A boolean indicating whether or not the first response category is 0

  # Returns:
  # A matrix representing the residuals for individual items

  theta_scores <- thetas[, 1]
  theta_scores[theta_scores == Inf] <- 10
  theta_scores[theta_scores == -Inf] <- -10
  residuals <- dplyr::as_tibble(matrix(0, nrow = nrow(data), ncol = ncol(data)))
  for (item in seq_len(ncol(data))) {
    mirt_item <- mirt::extract.item(model, item)
    item_probs <- mirt::probtrace(mirt_item, theta_scores)
    ifelse(start_0, indices <- data[, item] + 1, indices <- data[, item])
    score_probs <- mapply(function(row, col) item_probs[row, col], row = seq_len(nrow(item_probs)), col = indices)
    residuals[, item] <- 1 - score_probs
  }

  # Return a list containing all calculated residuals
  return(residuals)
}
