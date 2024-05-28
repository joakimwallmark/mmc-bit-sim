library(tidyverse)
library(arrow)
library(kableExtra)
library(patchwork)

py_result <- read_feather(paste0("simulation_results/py_swesat22b_nominal_quant.feather"))
r_result <- read_feather(paste0("simulation_results/r_swesat22b_nominal_quant.feather"))

# Combine R and python results --------------------------------------------
common_columns <- intersect(names(py_result), names(r_result))
unique_py <- setdiff(names(py_result), common_columns)
unique_r <- setdiff(names(r_result), common_columns)
# Add missing columns to both dataframes
for (col in unique_r) {
  py_result[[col]] <- NA
}
for (col in unique_py) {
  r_result[[col]] <- NA
}
combined_res <- rbind(py_result, r_result)
combined_res <- combined_res |> mutate(z_estimation_method = ifelse(is.na(z_estimation_method), "ML", z_estimation_method))

# Results table -----------------------------------------------------------
table_df <- combined_res |>
  mutate(z_estimation_method = ifelse(is.na(z_estimation_method), "ML", z_estimation_method)) |>
  filter(test_log_likelihood > -0.98) |>
  mutate(model = case_when(
    model == "NR" ~ "NR AE",
    model == "MMC" ~ "MMC AE",
    model == "mirt nominal" ~ "NR MML",
  )) |>
  select(iteration, n, items, model, test_log_likelihood, test_residuals, z_estimation_method) |>
  group_by(n, items, model, z_estimation_method) |>
  summarise(
    mean_test_log_likelihood = mean(test_log_likelihood),
    mean_test_residuals = mean(test_residuals),
    se_test_log_likelihood = sd(test_log_likelihood),
    se_test_residuals = sd(test_residuals)
  ) |>
  pivot_wider(
    names_from = z_estimation_method, values_from = c(mean_test_log_likelihood, mean_test_residuals, se_test_log_likelihood, se_test_residuals),
    names_sep = "_", # Separator between 'c' values and column names
    names_glue = "{z_estimation_method}_{.value}"
  ) |>
  mutate(
    `log-likelihood (ML)` = sprintf("%.4f (%.4f)", ML_mean_test_log_likelihood, ML_se_test_log_likelihood),
    `log-likelihood (NN)` = sprintf("%.4f (%.4f)", NN_mean_test_log_likelihood, NN_se_test_log_likelihood),
    `residuals (ML)` = sprintf("%.4f (%.4f)", ML_mean_test_residuals, ML_se_test_residuals),
    `residuals (NN)` = sprintf("%.4f (%.4f)", NN_mean_test_residuals, NN_se_test_residuals)
  ) |>
  select(n, items, model, `log-likelihood (ML)`, `log-likelihood (NN)`, `residuals (ML)`, `residuals (NN)`) |>
  mutate(across(everything(), ~ ifelse(is.na(.x) | .x == "NA (NA)", "-", .x))) |>
  ungroup()

latex_table <- table_df |>
  kable(format = "latex", booktabs = TRUE, caption = "Simulation results.", label = "simulation_results") %>%
  kable_styling(font_size = 7)

latex_table <- gsub("\\\\addlinespace", "", latex_table)
latex_table


# Figure ------------------------------------------------------------------
figure_df <- combined_res |>
  filter(z_estimation_method == "ML", test_log_likelihood > -0.98) |>
  mutate(
    model = recode(
      model,
      "NR" = "NR AE",
      "MMC" = "MMC AE",
      "mirt nominal" = "NR MML"
    ),
    n = as.factor(n),
    items = as.factor(items) |>
      fct_recode("20 items" = "20", "40 items" = "40", "80 items" = "80")
  ) |>
  select(iteration, n, items, model, test_log_likelihood, test_residuals) |>
  group_by(n, items, model) |>
  summarise(
    mean_test_log_likelihood = mean(test_log_likelihood),
    mean_test_residuals = mean(test_residuals),
    se_test_log_likelihood = sd(test_log_likelihood),
    se_test_residuals = sd(test_residuals)
  ) |>
  mutate(across(everything(), ~ ifelse(is.na(.x) | .x == "NA (NA)", "-", .x))) |>
  ungroup()

colors <- c("black", "gray40", "gray60")
interaction_plot <- figure_df |>
  ggplot(aes(x = n, y = mean_test_log_likelihood, color = model)) +
  geom_point() +
  geom_line(aes(group = model), linetype = "dashed") +
  geom_errorbar(
    aes(
      ymin = mean_test_log_likelihood - se_test_log_likelihood,
      ymax = mean_test_log_likelihood + se_test_log_likelihood
    ),
    width = 0.3,
    linewidth = 0.2
  ) +
  theme_bw() +
  labs(x = "Sample size", y = "Average log-likelihood") +
  theme(legend.position = "bottom") +
  scale_color_manual(values = colors) +
  facet_wrap(~items)

res_interaction_plot <- figure_df |>
  ggplot(aes(x = n, y = mean_test_residuals, color = model)) +
  geom_point() +
  geom_line(aes(group = model), linetype = "dashed") +
  geom_errorbar(
    aes(
      ymin = mean_test_residuals - se_test_residuals,
      ymax = mean_test_residuals + se_test_residuals
    ),
    width = 0.3,
    linewidth = 0.2
  ) +
  theme_bw() +
  labs(x = "Sample size", y = "Average residual") +
  theme(legend.position = "bottom") +
  scale_color_manual(values = colors) +
  guides(color = guide_legend(title = NULL)) +
  facet_wrap(~items)

interaction_plot <- interaction_plot + theme(legend.position = "none")
combined_plots <- interaction_plot / res_interaction_plot
print(combined_plots)
ggsave("simulation_results/interaction.eps", plot = combined_plots, width = 6.27, height = 8, units = "in", dpi = 300)
