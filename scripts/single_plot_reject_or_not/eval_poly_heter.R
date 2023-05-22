# eval poly and heter model -----------------------------------------------

library(tensorflow)
library(keras)
library(tidyverse)

poly_model <- keras$models$load_model(here::here("models/single_plot_poly"))
heter_model <- keras$models$load_model(here::here("models/single_plot_heter"))

default_dat_gen <- image_data_generator()

poly_test_gen <- default_dat_gen$flow_from_directory(here::here("data/single_plot/poly_test"),
                                                     target_size = c(224L, 224L),
                                                     batch_size = 32L,
                                                     shuffle = FALSE,
                                                     save_to_dir = FALSE,
                                                     classes = c("not_reject", "reject"))

heter_train_gen <- default_dat_gen$flow_from_directory(here::here("data/single_plot/heter_train"),
                                                       target_size = c(224L, 224L),
                                                       batch_size = 32L,
                                                       shuffle = FALSE,
                                                       save_to_dir = FALSE,
                                                       classes = c("not_reject", "reject"))

heter_test_gen <- default_dat_gen$flow_from_directory(here::here("data/single_plot/heter_test"),
                                                      target_size = c(224L, 224L),
                                                      batch_size = 32L,
                                                      shuffle = FALSE,
                                                      save_to_dir = FALSE,
                                                      classes = c("not_reject", "reject"))

poly_train_gen <- default_dat_gen$flow_from_directory(here::here("data/single_plot/poly_train"),
                                                      target_size = c(224L, 224L),
                                                      batch_size = 32L,
                                                      shuffle = FALSE,
                                                      save_to_dir = FALSE,
                                                      classes = c("not_reject", "reject"))

poly_performance <- poly_model$evaluate(poly_test_gen)
poly_performance_on_heter <- poly_model$evaluate(heter_train_gen)

heter_performance <- heter_model$evaluate(heter_test_gen)
heter_performance_on_poly <- heter_model$evaluate(poly_train_gen)

poly_pred <- poly_model$predict(poly_test_gen) %>%
  as.data.frame() %>%
  mutate(V3 = ifelse(V1 > V2, 0, 1)) %>%
  pull(V3)

poly_pred_on_heter <- poly_model$predict(heter_train_gen) %>%
  as.data.frame() %>%
  mutate(V3 = ifelse(V1 > V2, 0, 1)) %>%
  pull(V3)

heter_pred <- heter_model$predict(heter_test_gen) %>%
  as.data.frame() %>%
  mutate(V3 = ifelse(V1 > V2, 0, 1)) %>%
  pull(V3)

heter_pred_on_poly <- heter_model$predict(poly_train_gen) %>%
  as.data.frame() %>%
  mutate(V3 = ifelse(V1 > V2, 0, 1)) %>%
  pull(V3)

# Conf mat

yardstick::conf_mat(data.frame(truth = factor(poly_test_gen$labels),
                               estimate = factor(poly_pred)),
                    truth = truth,
                    estimate = estimate)

yardstick::conf_mat(data.frame(truth = factor(heter_test_gen$labels),
                               estimate = factor(heter_pred)),
                    truth = truth,
                    estimate = estimate)

yardstick::conf_mat(data.frame(truth = factor(heter_train_gen$labels),
                               estimate = factor(poly_pred_on_heter)),
                    truth = truth,
                    estimate = estimate)

yardstick::conf_mat(data.frame(truth = factor(poly_train_gen$labels),
                               estimate = factor(heter_pred_on_poly)),
                    truth = truth,
                    estimate = estimate)

# Bal acc

yardstick::bal_accuracy(data.frame(truth = factor(poly_test_gen$labels),
                                   estimate = factor(poly_pred)),
                        truth = truth,
                        estimate = estimate)

yardstick::bal_accuracy(data.frame(truth = factor(heter_test_gen$labels),
                                   estimate = factor(heter_pred)),
                        truth = truth,
                        estimate = estimate)

yardstick::bal_accuracy(data.frame(truth = factor(heter_train_gen$labels),
                                   estimate = factor(poly_pred_on_heter)),
                        truth = truth,
                        estimate = estimate)

yardstick::bal_accuracy(data.frame(truth = factor(poly_train_gen$labels),
                                   estimate = factor(heter_pred_on_poly)),
                        truth = truth,
                        estimate = estimate)

# roc auc

