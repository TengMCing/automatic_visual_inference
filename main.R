# README ------------------------------------------------------------------

# This is the main script to setup data, train computer vision models, etc.

# single residual plot model ----------------------------------------------

# Setup data --------------------------------------------------------------

library(tidyverse)
library(visage)

# Create paths
if (!dir.exists(here::here("data"))) dir.create(here::here("data"))
if (!dir.exists(here::here("data/shared"))) dir.create(here::here("data/shared"))

# Cache lineup data
if (file.exists(here::here("data/shared/vi_lineup.rds"))) {
  vi_lineup <- readRDS(here::here("data/shared/vi_lineup.rds"))
} else {
  vi_lineup <- get_vi_lineup()
  saveRDS(vi_lineup, file = here::here("data/shared/vi_lineup.rds"))
}

# Calculate p-values for plots from null lineups
null_lineup_p_values <- vi_survey %>%
  filter(null_lineup) %>%
  group_by(unique_lineup_id) %>%
  summarise(plot_id = 1:20,
            p_value = map_dbl(1:20, function(i) {
              current_detect <- map_lgl(str_split(selection, "_"), ~as.character(i) %in% .x)
              calc_p_value(current_detect, num_selection, alpha = alpha[1])
              })) %>%
  ungroup()

# Find all plots that would be rejected by the visual test
# This includes all rejected data plot,
# and rejected null plots from null lineups 
rejected_plots <- null_lineup_p_values %>%
  filter(p_value <= 0.05) %>%
  bind_rows(vi_survey %>%
              filter(!null_lineup, !attention_check) %>%
              filter(p_value <= 0.05) %>%
              group_by(unique_lineup_id) %>%
              summarise(plot_id = answer[1],
                        p_value = p_value[1]))

poly_rejected_plots <- rejected_plots %>%
  filter(grepl("poly", unique_lineup_id))

heter_rejected_plots <- rejected_plots %>%
  filter(grepl("heter", unique_lineup_id))

# Find all plots that would not be rejected by the visual test
# This includes all not rejected data plot,
# and not rejected null plots from null lineups
not_rejected_plots <- null_lineup_p_values %>%
  filter(p_value > 0.05) %>%
  bind_rows(vi_survey %>%
              filter(!null_lineup, !attention_check) %>%
              filter(p_value > 0.05) %>%
              group_by(unique_lineup_id) %>%
              summarise(plot_id = answer[1],
                        p_value = p_value[1]))

poly_not_rejected_plots <- not_rejected_plots
heter_not_rejected_plots <- not_rejected_plots

# Draw all single plots
draw_single_plots <- function(plots, folder = "reject") {
  plots_to_save <- plots %>%
    mutate(p = map2(unique_lineup_id, plot_id, function(ulid, pid) {
      
      # Get the original scale as it would be plotted in a lineup
      sample_lineup <- vi_lineup[[ulid]]$data %>%
        VI_MODEL$plot_lineup(theme = theme_light(), 
                             remove_axis = TRUE, 
                             remove_legend = TRUE, 
                             remove_grid_line = TRUE)
      
      # Generate ggplot with smaller base size to keep the point size consistent
      vi_lineup[[ulid]]$data %>%
        filter(k == pid) %>%
        VI_MODEL$plot(theme = theme_light(base_size = 11/5), 
                      remove_axis = TRUE, 
                      remove_legend = TRUE, 
                      remove_grid_line = TRUE) +
        xlim(layer_scales(sample_lineup)$x$range$range) +
        ylim(layer_scales(sample_lineup)$y$range$range)
    })) %>%
    pull(p)
  
  new_id <- 0
  for (this_plot in plots_to_save) {
    new_id <- new_id + 1
    
    # The lineup layout contains 4 rows and 5 cols
    ggsave(glue::glue(here::here("data/single_plot/{folder}/{new_id}.png")), 
           this_plot, 
           width = 7/5, 
           height = 7/4)
  }
}

# Split data into training and testing
set.seed(10086)
poly_train_reject_ids <- sample(nrow(poly_rejected_plots),
                                size = as.integer(0.8 * nrow(poly_rejected_plots)))
poly_train_rejected_plots <- poly_rejected_plots[poly_train_reject_ids, ]
poly_test_rejected_plots <- poly_rejected_plots[-poly_train_reject_ids, ]

poly_train_not_reject_ids <- sample(nrow(poly_not_rejected_plots),
                                    size = as.integer(0.8 * nrow(poly_not_rejected_plots)))
poly_train_not_rejected_plots <- poly_not_rejected_plots[poly_train_not_reject_ids, ]
poly_test_not_rejected_plots <- poly_not_rejected_plots[-poly_train_not_reject_ids, ]

heter_train_reject_ids <- sample(nrow(heter_rejected_plots),
                                 size = as.integer(0.8 * nrow(heter_rejected_plots)))
heter_train_rejected_plots <- heter_rejected_plots[heter_train_reject_ids, ]
heter_test_rejected_plots <- heter_rejected_plots[-heter_train_reject_ids, ]

heter_train_not_reject_ids <- sample(nrow(heter_not_rejected_plots),
                                     size = as.integer(0.8 * nrow(heter_not_rejected_plots)))
heter_train_not_rejected_plots <- heter_not_rejected_plots[heter_train_not_reject_ids, ]
heter_test_not_rejected_plots <- heter_not_rejected_plots[-heter_train_not_reject_ids, ]

draw_single_plots(poly_train_rejected_plots, folder = "poly_train/reject")
draw_single_plots(poly_test_rejected_plots, folder = "poly_test/reject")
draw_single_plots(poly_train_not_rejected_plots, folder = "poly_train/not_reject")
draw_single_plots(poly_test_not_rejected_plots, folder = "poly_test/not_reject")

draw_single_plots(heter_train_rejected_plots, folder = "heter_train/reject")
draw_single_plots(heter_test_rejected_plots, folder = "heter_test/reject")
draw_single_plots(heter_train_not_rejected_plots, folder = "heter_train/not_reject")
draw_single_plots(heter_test_not_rejected_plots, folder = "heter_test/not_reject")
