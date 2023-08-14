library(tensorflow)
library(keras)

# mixed, poly, heter: null or not_null
# multiclass: null, poly, heter, non_normal

mixed_mod <- load_model_tf(here::here("models/single_plot_reject_or_not/mixed"))

mixed_test_set <- flow_images_from_directory(here::here("data/single_plot_reject_or_not/mixed/test"), 
                                             target_size = c(224L, 224L),
                                             shuffle = FALSE)

# show plots --------------------------------------------------------------

show_plot <- function(plot_uid, folder = here::here("data")) {
  imager::load.image(system(glue::glue("find {folder} -name '{plot_uid}.png'"), intern = TRUE)[1]) %>%
    plot()
}

# test performance --------------------------------------------------------

library(yardstick)
library(tidyverse)
library(visage)

# mixed -------------------------------------------------------------------

mixed_test_pred <- as_tibble(as.data.frame(mixed_mod$predict(mixed_test_set)))
names(mixed_test_pred) <- names(mixed_test_set$class_indices)
mixed_test_pred <- mixed_test_pred %>%
  mutate(truth = names(mixed_test_set$class_indices)[c(mixed_test_set$classes) + 1]) %>%
  mutate(pred = names(mixed_test_set$class_indices)[(reject > not_reject) + 1]) %>%
  mutate(truth = factor(truth), pred = factor(pred))

accuracy(mixed_test_pred, 
         truth = truth, 
         estimate = pred)
bal_accuracy(mixed_test_pred, 
             truth = truth, 
             estimate = pred)
conf_mat(mixed_test_pred,
         truth = truth,
         estimate = pred)

bind_rows(mutate(mixed_test_pred, type = "mixed")) %>%
  group_by(type) %>%
  roc_curve(truth = truth, not_reject) %>%
  autoplot()

mixed_test_pred <- mixed_test_pred %>%
  mutate(unique_lineup_id = gsub(".*/(.*)_(.*).png", "\\1", mixed_test_set$filenames)) %>%
  mutate(plot = gsub(".*/(.*)_(.*).png", "\\2", mixed_test_set$filenames)) %>%
  mutate(plot = as.integer(plot)) %>%
  left_join(vi_survey %>% group_by(unique_lineup_id) %>% summarise(across(everything(), first))) %>%
  mutate(effect_size = ifelse(answer == plot, effect_size, 0))

mixed_test_pred %>%
  mutate(wrong = pred != truth) %>%
  mutate(wrong = c("Same", "Different")[wrong + 1]) %>%
  ggplot() +
  ggbeeswarm::geom_quasirandom(aes(effect_size, pred, col = wrong), groupOnX = FALSE) +
  facet_wrap(~type, ncol = 1, scales = "free_x") +
  labs(col = "") +
  scale_x_sqrt() +
  ylab("Prediction")

mixed_test_pred %>%
  mutate(wrong = pred != truth) %>%
  mutate(wrong = c("Same", "Different")[wrong + 1]) %>%
  filter(type == "polynomial") %>%
  filter(wrong == "Different") %>%
  filter(pred == "not_reject") %>%
  mutate(filename = paste0(unique_lineup_id, "_", plot)) %>%
  pull(filename) %>%
  map(~show_plot(.x))
