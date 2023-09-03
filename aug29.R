library(tensorflow)
library(keras)
library(tidyverse)
library(yardstick)
library(visage)

load_image <- function(image_path, target_size = c(224L, 224L)) {
  this_image <- image_load(image_path, target_size = target_size)
  this_input_array <- image_to_array(this_image)
  array_reshape(this_input_array, c(1L, 224L, 224L, 3L))
}

predict_image <- function(model, image_path, target_size = c(224L, 224L)) {
  this_image <- image_load(image_path, target_size = target_size)
  this_input_array <- image_to_array(this_image)
  model$predict(array_reshape(this_input_array, c(1L, 224L, 224L, 3L)))
}

plot_image <- function(plot_uid, folder = here::here("data/single_plot_null_or_not_sim_only/")) {
  imager::load.image(system(glue::glue("find {folder} -name '{plot_uid}.png'"), intern = TRUE)[1]) %>%
    plot(axes = FALSE)
}

predict_test_set <- function(model, test_folder, classes = NULL, target_size = c(224L, 224L)) {
  
  test_set <- flow_images_from_directory(test_folder, 
                                         target_size = target_size,
                                         shuffle = FALSE)
  test_pred <- as_tibble(as.data.frame(model$predict(test_set)))
  
  if (is.null(classes)) classes <- names(test_set$class_indices)
  names(test_pred) <- classes
  
  test_pred %>%
    mutate(pred = (function(dat){
      col_names <- names(dat)
      result <- c()
      for (i in 1:nrow(dat)) {
        row <- dat[i, ]
        result[i] <- col_names[which.max(row)]
      }
      result
    })(.)) %>%
    mutate(truth = names(test_set$class_indices)[c(test_set$classes) + 1]) %>%
    mutate(filename = test_set$filenames) %>%
    mutate(pred = factor(pred), truth = factor(truth))
}

predict_human_data_reject_or_not <- function(model, 
                               classes = c("reject", "not_reject"), 
                               target_size = c(224L, 224L)) {
  bind_rows(predict_test_set(model, 
                             here::here("data/single_plot_reject_or_not/mixed/train"), 
                             classes = classes),
            predict_test_set(model, 
                             here::here("data/single_plot_reject_or_not/mixed/test"), 
                             classes = classes))
}


predict_human_data_null_or_not <- function(model, 
                               target_size = c(224L, 224L)) {
  bind_rows(predict_test_set(model, 
                             here::here("data/shared/experiments/heteroskedasticity"), 
                             classes = classes),
            predict_test_set(model, 
                             here::here("data/shared/experiments/polynomial"), 
                             classes = classes))
}

poly_mod <- load_model_tf(here::here("models/single_plot_null_or_not_sim_only/poly"))
heter_mod <- load_model_tf(here::here("models/single_plot_null_or_not_sim_only/heter"))
mixed_mod <- load_model_tf(here::here("hyperparameter_tuning/models/single_plot_null_or_not_sim_only/mixed"))
human_mod <- load_model_tf(here::here("models/single_plot_reject_or_not/mixed"))

poly_test <- predict_test_set(poly_mod, here::here("data/single_plot_null_or_not_sim_only/poly/test"))
heter_test <- predict_test_set(heter_mod, here::here("data/single_plot_null_or_not_sim_only/heter/test"))
mixed_test <- predict_test_set(mixed_mod, here::here("data/single_plot_null_or_not_sim_only/mixed/test"))
human_test <- predict_test_set(human_mod, here::here("data/single_plot_reject_or_not/mixed/test"))

bal_accuracy(poly_test, truth = truth, estimate = pred)
bal_accuracy(heter_test, truth = truth, estimate = pred)
bal_accuracy(mixed_test, truth = truth, estimate = pred)
bal_accuracy(human_test, truth = truth, estimate = pred)


conf_mat(poly_test, truth = truth, estimate = pred)
conf_mat(heter_test, truth = truth, estimate = pred)
conf_mat(mixed_test, truth = truth, estimate = pred)


bind_rows(mutate(mixed_test, type = "mixed"), 
          mutate(poly_test, type = "poly"),
          mutate(heter_test, type = "heter")) %>%
  group_by(type) %>%
  roc_curve(truth = truth, not_null) %>%
  autoplot()

heter_on_poly <- predict_test_set(heter_mod, here::here("data/single_plot_null_or_not_sim_only/poly/test"))
mixed_on_poly <- predict_test_set(mixed_mod, here::here("data/single_plot_null_or_not_sim_only/poly/test"))

bal_accuracy(heter_on_poly, truth = truth, estimate = pred)
bal_accuracy(mixed_on_poly, truth = truth, estimate = pred)

conf_mat(heter_on_poly, truth = truth, estimate = pred)
conf_mat(mixed_on_poly, truth = truth, estimate = pred)

poly_on_heter <- predict_test_set(poly_mod, here::here("data/single_plot_null_or_not_sim_only/heter/test"))
mixed_on_heter <- predict_test_set(mixed_mod, here::here("data/single_plot_null_or_not_sim_only/heter/test"))

bal_accuracy(poly_on_heter, truth = truth, estimate = pred)
bal_accuracy(mixed_on_heter, truth = truth, estimate = pred)

conf_mat(poly_on_heter, truth = truth, estimate = pred)
conf_mat(mixed_on_heter, truth = truth, estimate = pred)

poly_on_non_normal <- predict_test_set(poly_mod, here::here("data/single_plot_null_or_not_sim_only/non_normal/test"))
heter_on_non_normal <- predict_test_set(heter_mod, here::here("data/single_plot_null_or_not_sim_only/non_normal/test"))
mixed_on_non_normal <- predict_test_set(mixed_mod, here::here("data/single_plot_null_or_not_sim_only/non_normal/test"))

bal_accuracy(poly_on_non_normal, truth = truth, estimate = pred)
bal_accuracy(heter_on_non_normal, truth = truth, estimate = pred)
bal_accuracy(mixed_on_non_normal, truth = truth, estimate = pred)

conf_mat(poly_on_non_normal, truth = truth, estimate = pred)
conf_mat(heter_on_non_normal, truth = truth, estimate = pred)
conf_mat(mixed_on_non_normal, truth = truth, estimate = pred)


poly_on_ar <- predict_test_set(poly_mod, here::here("data/single_plot_null_or_not_sim_only/ar1/test"))
heter_on_ar <- predict_test_set(heter_mod, here::here("data/single_plot_null_or_not_sim_only/ar1/test"))
mixed_on_ar <- predict_test_set(mixed_mod, here::here("data/single_plot_null_or_not_sim_only/ar1/test"))

bal_accuracy(poly_on_ar, truth = truth, estimate = pred)
bal_accuracy(heter_on_ar, truth = truth, estimate = pred)
bal_accuracy(mixed_on_ar, truth = truth, estimate = pred)

conf_mat(poly_on_ar, truth = truth, estimate = pred)
conf_mat(heter_on_ar, truth = truth, estimate = pred)
conf_mat(mixed_on_ar, truth = truth, estimate = pred)

poly_on_human <- predict_human_data_reject_or_not(poly_mod, classes = c("not_null", "null")) %>%
  mutate(pred = ifelse(pred == "not_null", "reject", "not_reject")) %>%
  mutate(pred = factor(pred), truth = factor(truth))
heter_on_human <- predict_human_data_reject_or_not(heter_mod, classes = c("not_null", "null")) %>%
  mutate(pred = ifelse(pred == "not_null", "reject", "not_reject")) %>%
  mutate(pred = factor(pred), truth = factor(truth))
mixed_on_human <- predict_human_data_reject_or_not(mixed_mod, classes = c("not_null", "null")) %>%
  mutate(pred = ifelse(pred == "not_null", "reject", "not_reject")) %>%
  mutate(pred = factor(pred), truth = factor(truth))

bal_accuracy(poly_on_human, truth = truth, estimate = pred)
bal_accuracy(heter_on_human, truth = truth, estimate = pred)
bal_accuracy(mixed_on_human, truth = truth, estimate = pred)
bal_accuracy(human_test, truth = truth, estimate = pred)

conf_mat(poly_on_human, truth = truth, estimate = pred)
conf_mat(heter_on_human, truth = truth, estimate = pred)
conf_mat(mixed_on_human, truth = truth, estimate = pred)
conf_mat(human_test, truth = truth, estimate = pred)


bind_rows(mutate(poly_on_human, type = "poly", not_reject = null), 
          mutate(heter_on_human, type = "heter", not_reject = null),
          mutate(mixed_on_human, type = "mixed", not_reject = null),
          mutate(human_test, type = "human")) %>%
  group_by(type) %>%
  roc_curve(truth = truth, not_reject) %>%
  autoplot()


# human_data_null_or_not --------------------------------------------------

poly_on_human_null <- predict_human_data_null_or_not(poly_mod)
heter_on_human_null <- predict_human_data_null_or_not(heter_mod)
mixed_on_human_null <- predict_human_data_null_or_not(mixed_mod)


mixed_on_human_null %>%
  mutate(unique_lineup_id = gsub(".*/(.*_.*)_(.*).png", "\\1", filename)) %>%
  mutate(plot = gsub(".*/(.*_.*)_(.*).png", "\\2", filename)) %>%
  left_join(vi_survey %>% 
              select(unique_lineup_id, 
                     attention_check, 
                     null_lineup, 
                     answer, 
                     effect_size,
                     conventional_p_value,
                     p_value,
                     type,
                     shape,
                     a,
                     b,
                     x_dist,
                     e_dist,
                     e_sigma,
                     include_z,
                     n,
                     alpha) %>%
              group_by(unique_lineup_id) %>%
              slice_head(n = 1)) %>%
  group_by(unique_lineup_id) %>%
  mutate(rank = rank(not_null, ties.method = "min")) %>%
  ungroup() %>%
  filter(plot == answer) %>%
  mutate(model_reject = rank == 1) %>%
  mutate(diff_decision = (p_value <= 0.05) != model_reject) %>%
  ggplot() +
  ggbeeswarm::geom_quasirandom(aes(effect_size, model_reject, col = diff_decision),
                               orientation = "y",
                               alpha = 0.6) +
  facet_grid(x_dist~type, scales = "free_x") +
  scale_x_sqrt()

  mutate(answer = map_dbl(unique_lineup_id, 
                          ~vi_survey %>% 
                            filter(unique_lineup_id == .x) %>%
                            pull(answer) %>%
                            .[1])) %>%
  View()

# heatmap -----------------------------------------------------------------



all_heatmaps <- tibble()

for (file in list.files(here::here("gradcam/single_plot_null_or_not_sim_only/mixed/test/not_null"))) {
  all_heatmaps <- bind_rows(all_heatmaps,
                            read_csv(here::here(glue::glue("gradcam/single_plot_null_or_not_sim_only/mixed/test/not_null/{file}")), 
                                col_names = FALSE) %>%
                              mutate(row = 1:14, file = file))
}

for (file in list.files(here::here("gradcam/single_plot_null_or_not_sim_only/mixed/test/null"))) {
  all_heatmaps <- bind_rows(all_heatmaps,
                            read_csv(here::here(glue::glue("gradcam/single_plot_null_or_not_sim_only/mixed/test/null/{file}")), 
                                     col_names = FALSE) %>%
                              mutate(row = 1:14, file = file))
}

saveRDS(all_heatmaps, "heatmap.rds")

meta <- readRDS(here::here("data/single_plot_null_or_not_sim_only/meta.rds"))

all_heatmaps<- all_heatmaps %>%
  mutate(pred = gsub(".*_([01]).csv", "\\1", file)) %>%
  %>% mutate(pred = as.integer(pred)) %>%
  mutate(plot_uid =  gsub("_[01].csv", "", file)) %>%
  mutate(plot_uid = as.integer(plot_uid)) %>%
  left_join(meta)

plot_heatmap <- function(dat) {
  pivot_longer(dat, X1:X14) %>%
    mutate(name = gsub("X", "", name)) %>%
    mutate(name = as.integer(name)) %>%
    ggplot() +
    geom_tile(aes(name, row, fill = value)) +
    xlab("") +
    ylab("") +
    theme_minimal() +
    theme(axis.title = element_blank(),
          axis.ticks = element_blank(),
          axis.line = element_blank(),
          axis.text = element_blank(),
          line = element_blank()) +
    coord_fixed() +
    scale_fill_viridis_c(option = "B", limits = c(-0.01, 0.05))
}



all_heatmaps %>%
  filter(a == 0) %>%
  filter(b == 64) %>%
  filter(x_dist == "uniform") %>%
  filter(pred == 0) %>%
  group_by(row) %>%
  summarise(across(X1:X14, mean)) %>%
  plot_heatmap()
