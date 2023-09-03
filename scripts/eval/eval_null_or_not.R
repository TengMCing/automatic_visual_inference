library(tensorflow)
library(keras)

# mixed, poly, heter: null or not_null
# multiclass: null, poly, heter, non_normal

mixed_mod <- load_model_tf(here::here("models/single_plot_null_or_not_sim_only/mixed"))
poly_mod <- load_model_tf(here::here("models/single_plot_null_or_not_sim_only/poly"))
heter_mod <- load_model_tf(here::here("models/single_plot_null_or_not_sim_only/heter"))
multiclass_mod <- load_model_tf(here::here("models/single_plot_null_or_not_sim_only/mixed_multiclass"))

mixed_test_set <- flow_images_from_directory(here::here("data/single_plot_null_or_not_sim_only/mixed/test"), 
                                             target_size = c(224L, 224L),
                                             shuffle = FALSE)
poly_test_set <- flow_images_from_directory(here::here("data/single_plot_null_or_not_sim_only/poly/test"), 
                                            target_size = c(224L, 224L),
                                            shuffle = FALSE)
heter_test_set <- flow_images_from_directory(here::here("data/single_plot_null_or_not_sim_only/heter/test"), 
                                             target_size = c(224L, 224L),
                                             shuffle = FALSE)
multiclass_test_set <- flow_images_from_directory(here::here("data/single_plot_null_or_not_sim_only/mixed_multiclass/test"), 
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

meta_data <- readRDS(here::here("data/single_plot_null_or_not_sim_only/meta.rds")) %>%
  as_tibble()


# mixed -------------------------------------------------------------------

mixed_test_pred <- as_tibble(as.data.frame(mixed_mod$predict(mixed_test_set)))
names(mixed_test_pred) <- names(mixed_test_set$class_indices)
mixed_test_pred <- mixed_test_pred %>%
  mutate(truth = names(mixed_test_set$class_indices)[c(mixed_test_set$classes) + 1]) %>%
  mutate(pred = names(mixed_test_set$class_indices)[(null > not_null) + 1]) %>%
  mutate(truth = factor(truth), pred = factor(pred))

accuracy(mixed_test_pred, 
         truth = truth, 
         estimate = pred)
conf_mat(mixed_test_pred,
         truth = truth,
         estimate = pred)


# poly --------------------------------------------------------------------

poly_test_pred <- as_tibble(as.data.frame(poly_mod$predict(poly_test_set)))
names(poly_test_pred) <- names(poly_test_set$class_indices)
poly_test_pred <- poly_test_pred %>%
  mutate(truth = names(poly_test_set$class_indices)[c(poly_test_set$classes) + 1]) %>%
  mutate(pred = names(poly_test_set$class_indices)[(null > not_null) + 1]) %>%
  mutate(truth = factor(truth), pred = factor(pred))

accuracy(poly_test_pred, 
         truth = truth, 
         estimate = pred)
conf_mat(poly_test_pred,
         truth = truth,
         estimate = pred)


# heter -------------------------------------------------------------------

heter_test_pred <- as_tibble(as.data.frame(heter_mod$predict(heter_test_set)))
names(heter_test_pred) <- names(heter_test_set$class_indices)
heter_test_pred <- heter_test_pred %>%
  mutate(truth = names(heter_test_set$class_indices)[c(heter_test_set$classes) + 1]) %>%
  mutate(pred = names(heter_test_set$class_indices)[(null > not_null) + 1]) %>%
  mutate(truth = factor(truth), pred = factor(pred))

accuracy(heter_test_pred, 
         truth = truth, 
         estimate = pred)
conf_mat(heter_test_pred,
         truth = truth,
         estimate = pred)


# roc ---------------------------------------------------------------------

bind_rows(mutate(mixed_test_pred, type = "mixed"), 
          mutate(poly_test_pred, type = "poly"),
          mutate(heter_test_pred, type = "heter")) %>%
  group_by(type) %>%
  roc_curve(truth = truth, not_null) %>%
  autoplot()


# multiclass --------------------------------------------------------------

multiclass_test_pred <- as_tibble(as.data.frame(multiclass_mod$predict(multiclass_test_set)))
names(multiclass_test_pred) <- names(multiclass_test_set$class_indices)
multiclass_test_pred <- multiclass_test_pred %>%
  mutate(pred = apply(multiclass_test_pred, 1, which.max)) %>%
  mutate(pred = names(multiclass_test_set$class_indices)[pred]) %>%
  mutate(truth = names(multiclass_test_set$class_indices)[c(multiclass_test_set$classes) + 1]) %>%
  mutate(truth = factor(truth), pred = factor(pred))

accuracy(multiclass_test_pred, 
         truth = truth, 
         estimate = pred)
bal_accuracy(multiclass_test_pred, 
             truth = truth, 
             estimate = pred)
conf_mat(multiclass_test_pred,
         truth = truth,
         estimate = pred)


# mixed detail ------------------------------------------------------------

mixed_test_detail <- mixed_test_pred %>%
  mutate(plot_uid = gsub(".*/(.*).png", "\\1", mixed_test_set$filenames) %>%
           as.numeric()) %>%
  left_join(meta_data)

mixed_test_detail %>%
  mutate(violation = ifelse(!is.na(shape), "poly", "other")) %>%
  mutate(violation = ifelse(!is.na(a), "heter", violation)) %>%
  mutate(violation = ifelse(!is.na(e_dist), "non_normal", violation)) %>%
  mutate(violation = ifelse(truth == "null", "null", violation)) %>%
  group_by(violation) %>%
  summarise(acc = mean(truth == pred), 
            correct = sum(truth == pred), 
            total = n())

mixed_test_detail %>%
  filter(truth == "not_null") %>%
  filter(!is.na(shape)) %>%
  ggplot() +
  geom_boxplot(aes(factor(e_sigma), not_null), outlier.alpha = 0) +
  geom_jitter(aes(factor(e_sigma), not_null), alpha = 0.3, height = 0, width = 0.2) +
  ggtitle("For residual plots with polynomial visual features", subtitle = "The predicted probability of not null is") +
  facet_wrap(~shape, labeller = label_both) +
  xlab("sigma of the error term") +
  ylab("Predicted probability")

mixed_test_detail %>%
  filter(truth == "not_null") %>%
  filter(!is.na(shape)) %>%
  filter(e_sigma == 0.5) %>%
  filter(not_null < 0.6) %>%
  pull(plot_uid)



mixed_test_detail %>%
  filter(truth == "not_null") %>%
  filter(!is.na(a)) %>%
  ggplot() +
  geom_boxplot(aes(factor(b), not_null), outlier.alpha = 0) +
  geom_jitter(aes(factor(b), not_null), alpha = 0.3, height = 0, width = 0.2) +
  ggtitle("For residual plots with heteroskedasticity visual features", subtitle = "The predicted probability of not null is") +
  facet_wrap(~a, labeller = label_both) +
  xlab("b") +
  ylab("Predicted probability")

mixed_test_detail %>%
  filter(truth == "not_null") %>%
  filter(!is.na(df)) %>%
  ggplot() +
  geom_boxplot(aes(factor(e_sigma), not_null), outlier.alpha = 0) +
  geom_jitter(aes(factor(e_sigma), not_null), alpha = 0.3, height = 0, width = 0.2) +
  ggtitle("For residual plots with non-normal visual features", subtitle = "The predicted probability of not null is") +
  facet_wrap(~e_dist) +
  xlab("sigma") +
  ylab("Predicted probability")

mixed_test_detail %>%
  filter(truth == "not_null") %>%
  filter(!is.na(df)) %>%
  filter(e_dist == "t") %>%
  ggplot() +
  geom_boxplot(aes(factor(df), not_null), outlier.alpha = 0) +
  geom_jitter(aes(factor(df), not_null), alpha = 0.3, height = 0, width = 0.2) +
  ggtitle("For residual plots with non-normal visual features", subtitle = "The predicted probability of not null is") +
  facet_wrap(~e_sigma, labeller = label_both) +
  xlab("degree of freedom") +
  ylab("Predicted probability")

# poly_on_heter -----------------------------------------------------------

poly_on_heter_pred <- as_tibble(as.data.frame(poly_mod$predict(heter_test_set)))
names(poly_on_heter_pred) <- names(heter_test_set$class_indices)
poly_on_heter_pred <- poly_on_heter_pred %>%
  mutate(truth = names(heter_test_set$class_indices)[c(heter_test_set$classes) + 1]) %>%
  mutate(pred = names(heter_test_set$class_indices)[(null > not_null) + 1]) %>%
  mutate(truth = factor(truth), pred = factor(pred))

accuracy(poly_on_heter_pred, 
         truth = truth, 
         estimate = pred)
conf_mat(poly_on_heter_pred,
         truth = truth,
         estimate = pred)


# heter_on_poly -----------------------------------------------------------


heter_on_poly_pred <- as_tibble(as.data.frame(heter_mod$predict(poly_test_set)))
names(heter_on_poly_pred) <- names(poly_test_set$class_indices)
heter_on_poly_pred <- heter_on_poly_pred %>%
  mutate(truth = names(poly_test_set$class_indices)[c(poly_test_set$classes) + 1]) %>%
  mutate(pred = names(poly_test_set$class_indices)[(null > not_null) + 1]) %>%
  mutate(truth = factor(truth), pred = factor(pred))

accuracy(heter_on_poly_pred, 
         truth = truth, 
         estimate = pred)
conf_mat(heter_on_poly_pred,
         truth = truth,
         estimate = pred)



# visual experiment -------------------------------------------------------

vi_lineup <- readRDS(here::here("data/shared/vi_lineup.rds"))
library(visage)

if (!file.exists(here::here("data/shared/experiments"))) {
  for (lineup in vi_lineup) {
    
    if (lineup$metadata$effect_size == 0) next
    
    this_plot <- lineup$data %>%
      filter(null == FALSE) %>%
      VI_MODEL$plot(theme = theme_light(base_size = 11/5), 
                    remove_axis = TRUE, 
                    remove_legend = TRUE, 
                    remove_grid_line = TRUE)
    
    pos <- lineup$data %>%
      filter(null == FALSE) %>%
      pull(k) %>%
      .[1]
    
    ggsave(glue::glue(here::here("data/shared/experiments/{lineup$metadata$type}/not_null/{lineup$metadata$name}_{pos}.png")), 
           this_plot, 
           width = 7/5, 
           height = 7/4)
  }
  
  for (lineup in vi_lineup) {
    
    for (i in 1:20) {
      
      null_flag <- lineup$data %>% 
        filter(k == i) %>% 
        pull(null) %>%
        .[1]
      
      if (lineup$metadata$effect_size > 0 && null_flag == FALSE) next
      
      this_plot <- lineup$data %>% 
        filter(k == i) %>%
        VI_MODEL$plot(theme = theme_light(base_size = 11/5), 
                      remove_axis = TRUE, 
                      remove_legend = TRUE, 
                      remove_grid_line = TRUE)
      
      ggsave(glue::glue(here::here("data/shared/experiments/{lineup$metadata$type}/null/{lineup$metadata$name}_{i}.png")), 
             this_plot, 
             width = 7/5, 
             height = 7/4)
    }
  }
}

visual_poly_test_set <- flow_images_from_directory(here::here("data/shared/experiments/polynomial"), 
                                                   target_size = c(224L, 224L),
                                                   shuffle = FALSE)

visual_poly_test_pred <- as_tibble(as.data.frame(mixed_mod$predict(visual_poly_test_set)))
names(visual_poly_test_pred) <- c("not_null", "null")
visual_poly_test_pred <- visual_poly_test_pred %>%
  mutate(truth = "not_null") %>%
  mutate(pred = c("not_null", "null")[(null > not_null) + 1]) %>%
  mutate(truth = factor(truth, levels = c("not_null", "null")), pred = factor(pred))

accuracy(visual_poly_test_pred, 
         truth = truth, 
         estimate = pred)
conf_mat(visual_poly_test_pred,
         truth = truth,
         estimate = pred)

visual_heter_test_set <- flow_images_from_directory(here::here("data/shared/experiments/heteroskedasticity"), 
                                                    target_size = c(224L, 224L),
                                                    shuffle = FALSE)

visual_heter_test_pred <- as_tibble(as.data.frame(mixed_mod$predict(visual_heter_test_set)))
names(visual_heter_test_pred) <- c("not_null", "null")
visual_heter_test_pred <- visual_heter_test_pred %>%
  mutate(truth = "not_null") %>%
  mutate(pred = c("not_null", "null")[(null > not_null) + 1]) %>%
  mutate(truth = factor(truth, levels = c("not_null", "null")), pred = factor(pred))

accuracy(visual_heter_test_pred, 
         truth = truth, 
         estimate = pred)
conf_mat(visual_heter_test_pred,
         truth = truth,
         estimate = pred)

visual_pred <- bind_rows(visual_poly_test_pred %>%
                           mutate(unique_lineup_id = gsub(".*/(.*).png", "\\1", visual_poly_test_set$filenames)),
                         visual_heter_test_pred %>%
                           mutate(unique_lineup_id = gsub(".*/(.*).png", "\\1", visual_heter_test_set$filenames)))

visual_pred <- vi_survey %>%
  group_by(unique_lineup_id) %>%
  summarise(across(everything(), first)) %>%
  right_join(visual_pred)

visual_pred %>%
  count(pred == "not_null", p_value <= 0.05)

jitter_pos <- position_jitter(width = 0)

library(ggbeeswarm)

visual_pred %>%
  mutate(diff_decision = (p_value <= 0.05) != (pred == "not_null")) %>%
  ggplot() +
  geom_quasirandom(aes(effect_size, pred == "not_null", col = diff_decision),
                       groupOnX = FALSE,
                       alpha = 0.6) +
  facet_wrap(~type, ncol = 1, scales = "free_x") +
  scale_x_log10()


