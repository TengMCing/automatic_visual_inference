default_repo <- "https://cloud.r-project.org"
if (!requireNamespace("haven", quietly = TRUE)) install.packages("haven", repos = default_repo)
if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse", repos = default_repo)
if (!requireNamespace("here", quietly = TRUE)) install.packages("here", repos = default_repo)
if (!requireNamespace("glue", quietly = TRUE)) install.packages("glue", repos = default_repo)
if (!requireNamespace("progress", quietly = TRUE)) install.packages("progress", repos = default_repo)
if (!requireNamespace("visage", quietly = TRUE)) {
  remotes::install_github("TengMCing/bandicoot")
  remotes::install_github("TengMCing/visage")
}


library(tidyverse)
library(visage)

set.seed(10086)

# shared data -------------------------------------------------------------

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


# progress bar ------------------------------------------------------------

new_pb <- function(violation, total) {
  barstr <- glue::glue("[:spin] {violation} parameter :what [:bar] :current/:total (:percent) eta: :eta :tick_rate/sec")
  pb <- progress::progress_bar$new(format = barstr,
                                   total = total,
                                   clear = FALSE,
                                   width = 60)
  return(pb)
}

update_pb <- function(pb, i) {
  pb$tick(token = list(what = i))
}

# Global setting ----------------------------------------------------------

# The global uid for plots
PLOT_UID <- 0

# The global meta data for plots
PLOT_META <- data.frame()

# Draw plots for a violation model
draw_plots <- function(violation, not_null, null, n, meta_vector) {
  mod <- list()
  mod$not_null <- not_null
  mod$null <- null
  
  for (response in c("not_null", "null")) {
    for (data_type in c("train", "test")) {
      plot_dat <- map(1:SAMPLE_PER_PARAMETER[[data_type]], 
                      ~mod[[response]]$gen_lineup(n))
      
      pos <- map_dbl(plot_dat, ~.x %>% filter(null == FALSE) %>% pull(k) %>% .[1])
      
      # Speed up the plot drawing
      num_plots <- length(plot_dat)
      
      map2(plot_dat, (PLOT_UID + 1):(PLOT_UID + num_plots), function(this_dat, this_plot_id) {
        this_plot <- this_dat %>%
          VI_MODEL$plot_lineup(theme = theme_light(), 
                               remove_axis = TRUE, 
                               remove_legend = TRUE, 
                               remove_grid_line = TRUE)
        
        # The lineup layout contains 4 rows and 5 cols
        ggsave(glue::glue(here::here("data/lineup_null_or_not_sim_only/{violation}/{data_type}/{response}/{this_plot_id}.png")), 
               this_plot, 
               width = 7, 
               height = 7)
      })
      
      for (i in 1:num_plots) {
        PLOT_UID <<- PLOT_UID + 1
        PLOT_META <<- PLOT_META %>%
          bind_rows(c(plot_uid = PLOT_UID, 
                      meta_vector, 
                      k = pos[i],
                      data_type = data_type, 
                      response = response))
      }
    }
  }
}

# Ensure the support of the predictor is [-1, 1]
stand_dist <- function(x) (x - min(x))/max(x - min(x)) * 2 - 1

SAMPLE_PER_PARAMETER <- list(train = 100, test = 10)

# Define the x variable
rand_uniform_x <- rand_uniform(-1, 1)
rand_normal_raw_x <- rand_normal(sigma = 0.3)
rand_normal_x <- closed_form(~stand_dist(rand_normal_raw_x))
rand_lognormal_raw_x <- rand_lognormal(sigma = 0.6)
rand_lognormal_x <- closed_form(~stand_dist(rand_lognormal_raw_x/3 - 1))
rand_discrete_x <- rand_uniform_d(-1, 1, k = 5, even = TRUE)

get_x_var <- function(dist_name) {
  switch(as.character(dist_name),
         uniform = rand_uniform_x,
         normal = rand_normal_x,
         lognormal = rand_lognormal_x,
         even_discrete = rand_discrete_x)
} 

# poly_data ---------------------------------------------------------------

model_parameters <- expand.grid(shape = 1:4,
                                e_sigma = c(0.5, 1, 2, 4),
                                x_dist = c("uniform", 
                                           "normal", 
                                           "lognormal", 
                                           "even_discrete"),
                                n = c(50, 100, 300))

pb <- new_pb("Poly", nrow(model_parameters))

for (i in 1:nrow(model_parameters)) {
  update_pb(pb, i)
  draw_plots(violation = "poly",
             not_null = poly_model(model_parameters$shape[i], 
                                   x = get_x_var(model_parameters$x_dist[i]), 
                                   sigma = model_parameters$e_sigma[i]),
             null = poly_model(model_parameters$shape[i], 
                               x = get_x_var(model_parameters$x_dist[i]), 
                               include_z = FALSE,
                               sigma = model_parameters$e_sigma[i]),
             n = model_parameters$n[i],
             meta_vector = c(model_parameters[i, ]))
  
}

# save_meta_data ----------------------------------------------------------

saveRDS(PLOT_META, here::here("data/lineup_null_or_not_sim_only/meta.rds"))

# heter_data --------------------------------------------------------------

model_parameters <- expand.grid(a = c(-1, 0, 1),
                                b = c(0.25, 1, 4, 16, 64),
                                x_dist = c("uniform", 
                                           "normal", 
                                           "lognormal", 
                                           "even_discrete"),
                                n = c(50, 100, 300))

pb <- new_pb("Heter", nrow(model_parameters))

for (i in 1:nrow(model_parameters)) {
  update_pb(pb, i)
  draw_plots(violation = "heter",
             not_null = heter_model(a = model_parameters$a[i],
                                    b = model_parameters$b[i],
                                    x = get_x_var(model_parameters$x_dist[i])),
             null = heter_model(a = model_parameters$a[i],
                                b = 0,
                                x = get_x_var(model_parameters$x_dist[i])),
             n = model_parameters$n[i],
             meta_vector = c(model_parameters[i, ]))
  
}

# save_meta_data ----------------------------------------------------------

saveRDS(PLOT_META, here::here("data/lineup_null_or_not_sim_only/meta.rds"))

# non_normal --------------------------------------------------------------

model_parameters <- expand.grid(x_dist = c("uniform", 
                                           "normal", 
                                           "lognormal", 
                                           "even_discrete"),
                                e_dist = c("uniform",
                                           "lognormal",
                                           "even_discrete",
                                           "t"),
                                df = c(0, 1, 2, 5, 10),
                                e_sigma = c(0.5, 1, 2, 4),
                                n = c(50, 100, 300))

model_parameters <- model_parameters %>%
  filter(!(e_dist != "t" & df != 0)) %>%
  filter(!(e_dist == "t" & df == 0))

lognormal_sigma_table <- map_dbl(seq(0.001, 2, 0.001), ~sqrt((exp(.x^2) - 1) * exp(.x^2)))
names(lognormal_sigma_table) <- seq(0.001, 2, 0.001)

get_e_var <- function(dist_name, df, e_sigma) {
  
  dist_name <- as.character(dist_name)
  
  if (dist_name == "uniform") {
    return(rand_uniform(a = -sqrt(12 * e_sigma^2)/2, 
                        b = sqrt(12 * e_sigma^2)/2,
                        env = new.env(parent = .GlobalEnv)))
  }
  
  if (dist_name == "lognormal") {
    table_index <- which.min(abs(lognormal_sigma_table - e_sigma))
    mod_sigma <- as.numeric(names(lognormal_sigma_table))[table_index]
    return(rand_lognormal(mu = 0,
                          sigma = mod_sigma,
                          env = new.env(parent = .GlobalEnv)))
  }
  
  if (dist_name == "even_discrete") {
    return(rand_uniform_d(a = -sqrt(12 * e_sigma^2)/2, 
                          b = sqrt(12 * e_sigma^2)/2,
                          even = TRUE,
                          env = new.env(parent = .GlobalEnv)))
  }
  
  if (dist_name == "t") {
    tau <- 1
    if (df > 2) tau <- sqrt(e_sigma^2 * (df - 2)/df)
    return(rand_t(tau = tau, 
                  df = df, 
                  env = new.env(parent = .GlobalEnv)))
  }
  
  return(rand_normal(sigma = e_sigma, 
                     env = new.env(parent = .GlobalEnv)))
}

pb <- new_pb("Non-normal", nrow(model_parameters))

for (i in 1:nrow(model_parameters)) {
  update_pb(pb, i)
  draw_plots(violation = "non_normal",
             not_null = non_normal_model(x = get_x_var(model_parameters$x_dist[i]),
                                         e = get_e_var(model_parameters$e_dist[i],
                                                       model_parameters$df[i],
                                                       model_parameters$e_sigma[i])),
             null = non_normal_model(x = get_x_var(model_parameters$x_dist[i]),
                                     e = get_e_var("normal", 
                                                   0, 
                                                   model_parameters$e_sigma[i])),
             n = model_parameters$n[i],
             meta_vector = c(model_parameters[i, ]))
  
}

# save_meta_data ----------------------------------------------------------

saveRDS(PLOT_META, here::here("data/lineup_null_or_not_sim_only/meta.rds"))

# mixed_data --------------------------------------------------------------

if (!dir.exists(here::here("data/lineup_null_or_not_sim_only/mixed"))) dir.create(here::here("data/lineup_null_or_not_sim_only/mixed"))
if (!dir.exists(here::here("data/lineup_null_or_not_sim_only/mixed/train"))) dir.create(here::here("data/lineup_null_or_not_sim_only/mixed/train"))
if (!dir.exists(here::here("data/lineup_null_or_not_sim_only/mixed/test"))) dir.create(here::here("data/lineup_null_or_not_sim_only/mixed/test"))

mixed_train_dest <- here::here("data/lineup_null_or_not_sim_only/mixed/train")
mixed_test_dest <- here::here("data/lineup_null_or_not_sim_only/mixed/test")
for (violation in c("poly", "heter", "non_normal")) {
  train_from <- here::here(glue::glue("data/lineup_null_or_not_sim_only/{violation}/train/."))
  test_from <- here::here(glue::glue("data/lineup_null_or_not_sim_only/{violation}/test/."))
  system(glue::glue("cp -r {train_from} {mixed_train_dest}"))
  system(glue::glue("cp -r {test_from} {mixed_test_dest}"))
}

# mixed_multiclass_data ---------------------------------------------------

if (!dir.exists(here::here("data/lineup_null_or_not_sim_only/mixed_multiclass"))) dir.create(here::here("data/lineup_null_or_not_sim_only/mixed_multiclass"))
if (!dir.exists(here::here("data/lineup_null_or_not_sim_only/mixed_multiclass/train"))) dir.create(here::here("data/lineup_null_or_not_sim_only/mixed_multiclass/train"))
if (!dir.exists(here::here("data/lineup_null_or_not_sim_only/mixed_multiclass/test"))) dir.create(here::here("data/lineup_null_or_not_sim_only/mixed_multiclass/test"))

for (violation in c("poly", "heter", "non_normal")) {
  train_not_null_from <- here::here(glue::glue("data/lineup_null_or_not_sim_only/{violation}/train/not_null/."))
  system(glue::glue("cp -r {train_not_null_from} {here::here(paste0('data/lineup_null_or_not_sim_only/mixed_multiclass/train/', violation))}"))
  
  test_not_null_from <- here::here(glue::glue("data/lineup_null_or_not_sim_only/{violation}/test/not_null/.")) 
  system(glue::glue("cp -r {test_not_null_from} {here::here(paste0('data/lineup_null_or_not_sim_only/mixed_multiclass/test/', violation))}"))
}


num_null_train_plots <- length(list.files("data/lineup_null_or_not_sim_only/non_normal/train/not_null/"))
num_null_test_plots <- length(list.files("data/lineup_null_or_not_sim_only/non_normal/test/not_null/"))

set.seed(10086)
train_null_ids <- sample(list.files(paste0(mixed_train_dest, "/null")), num_null_train_plots)
test_null_ids <- sample(list.files(paste0(mixed_test_dest, "/null")), num_null_test_plots)

train_null_from <- paste0(mixed_train_dest, "/null")
test_null_from <- paste0(mixed_test_dest, "/null")
if (!dir.exists(here::here("data/lineup_null_or_not_sim_only/mixed_multiclass/train/null"))) dir.create(here::here("data/lineup_null_or_not_sim_only/mixed_multiclass/train/null"))
if (!dir.exists(here::here("data/lineup_null_or_not_sim_only/mixed_multiclass/test/null"))) dir.create(here::here("data/lineup_null_or_not_sim_only/mixed_multiclass/test/null"))

for (id in train_null_ids) {
  system(glue::glue("cp {paste0(train_null_from, '/', id)} {here::here(paste0('data/lineup_null_or_not_sim_only/mixed_multiclass/train/null/', id))}"))
}

for (id in test_null_ids) {
  system(glue::glue("cp {paste0(test_null_from, '/', id)} {here::here(paste0('data/lineup_null_or_not_sim_only/mixed_multiclass/test/null/', id))}"))
}
