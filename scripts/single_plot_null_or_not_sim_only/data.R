library(tidyverse)
library(visage)

set.seed(10086)

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

draw_single_plots <- function(plot_dat, folder, start_id) {
  plots_to_save <- map(plot_dat, function(this_dat) {
      # Generate ggplot with smaller base size to keep the point size consistent
      this_dat %>%
        VI_MODEL$plot(theme = theme_light(base_size = 11/5), 
                      remove_axis = TRUE, 
                      remove_legend = TRUE, 
                      remove_grid_line = TRUE)
    })
  
  new_id <- 0
  for (this_plot in plots_to_save) {
    new_id <- new_id + 1
    
    # The lineup layout contains 4 rows and 5 cols
    ggsave(glue::glue(here::here("data/single_plot_null_or_not_sim_only/{folder}/{start_id + new_id - 1}.png")), 
           this_plot, 
           width = 7/5, 
           height = 7/4)
  }
}

# Ensure the support of the predictor is [-1, 1]
stand_dist <- function(x) (x - min(x))/max(x - min(x)) * 2 - 1


# poly_data ---------------------------------------------------------------

model_parameters <- expand.grid(shape = 1:4,
                                e_sigma = c(0.5, 1, 2, 4),
                                x_dist = c("uniform", 
                                           "normal", 
                                           "lognormal", 
                                           "even_discrete"),
                                n = c(50, 100, 300))

for (i in 1:nrow(model_parameters)) {
  x <- switch(as.character(model_parameters$x_dist[i]),
              uniform = rand_uniform(-1, 1),
              normal = {
                raw_x <- rand_normal(sigma = 0.3)
                closed_form(~stand_dist(raw_x))
              },
              lognormal = {
                raw_x <- rand_lognormal(sigma = 0.6)
                closed_form(~stand_dist(raw_x/3 - 1))
              },
              even_discrete = rand_uniform_d(k = 5, even = TRUE))
  
  mod <- poly_model(model_parameters$shape[i], 
                    x = x, 
                    sigma = model_parameters$e_sigma[i])
  
  plot_dat <- map(1:50, ~mod$gen(model_parameters$n[i]))
  
  draw_single_plots(plot_dat, "poly/train/not_null", i * 50)
}

for (i in 1:nrow(model_parameters)) {
  x <- switch(as.character(model_parameters$x_dist[i]),
              uniform = rand_uniform(-1, 1),
              normal = {
                raw_x <- rand_normal(sigma = 0.3)
                closed_form(~stand_dist(raw_x))
              },
              lognormal = {
                raw_x <- rand_lognormal(sigma = 0.6)
                closed_form(~stand_dist(raw_x/3 - 1))
              },
              even_discrete = rand_uniform_d(k = 5, even = TRUE))
  
  mod <- poly_model(model_parameters$shape[i], 
                    x = x, 
                    sigma = model_parameters$e_sigma[i])
  
  plot_dat <- map(1:5, ~mod$gen(model_parameters$n[i]))
  
  draw_single_plots(plot_dat, "poly/test/not_null", i * 5)
}

for (i in 1:nrow(model_parameters)) {
  x <- switch(as.character(model_parameters$x_dist[i]),
              uniform = rand_uniform(-1, 1),
              normal = {
                raw_x <- rand_normal(sigma = 0.3)
                closed_form(~stand_dist(raw_x))
              },
              lognormal = {
                raw_x <- rand_lognormal(sigma = 0.6)
                closed_form(~stand_dist(raw_x/3 - 1))
              },
              even_discrete = rand_uniform_d(k = 5, even = TRUE))
  
  mod <- poly_model(model_parameters$shape[i], 
                    x = x, 
                    include_z = FALSE, 
                    sigma = model_parameters$e_sigma[i])
  
  plot_dat <- map(1:50, ~mod$gen(model_parameters$n[i]))
  
  draw_single_plots(plot_dat, "poly/train/null", i * 50)
}

for (i in 1:nrow(model_parameters)) {
  x <- switch(as.character(model_parameters$x_dist[i]),
              uniform = rand_uniform(-1, 1),
              normal = {
                raw_x <- rand_normal(sigma = 0.3)
                closed_form(~stand_dist(raw_x))
              },
              lognormal = {
                raw_x <- rand_lognormal(sigma = 0.6)
                closed_form(~stand_dist(raw_x/3 - 1))
              },
              even_discrete = rand_uniform_d(k = 5, even = TRUE))
  
  mod <- poly_model(model_parameters$shape[i], 
                    x = x, 
                    include_z = FALSE, 
                    sigma = model_parameters$e_sigma[i])
  
  plot_dat <- map(1:5, ~mod$gen(model_parameters$n[i]))
  
  draw_single_plots(plot_dat, "poly/test/null", i * 5)
}



# heter_data --------------------------------------------------------------

model_parameters <- expand.grid(a = c(-1, 0, 1),
                                b = c(0.25, 1, 4, 16, 64),
                                x_dist = c("uniform", 
                                           "normal", 
                                           "lognormal", 
                                           "even_discrete"),
                                n = c(50, 100, 300))

for (i in 1:nrow(model_parameters)) {
  x <- switch(as.character(model_parameters$x_dist[i]),
              uniform = rand_uniform(-1, 1),
              normal = {
                raw_x <- rand_normal(sigma = 0.3)
                closed_form(~stand_dist(raw_x))
              },
              lognormal = {
                raw_x <- rand_lognormal(sigma = 0.6)
                closed_form(~stand_dist(raw_x/3 - 1))
              },
              even_discrete = rand_uniform_d(k = 5, even = TRUE))
  
  mod <- heter_model(a = model_parameters$a[i],
                     b = model_parameters$b[i],
                     x = x)
  
  heter_dat <- map(1:50, ~mod$gen(model_parameters$n[i]))
  
  draw_single_plots(heter_dat, "heter/train/not_null", i * 50)
}

for (i in 1:nrow(model_parameters)) {
  x <- switch(as.character(model_parameters$x_dist[i]),
              uniform = rand_uniform(-1, 1),
              normal = {
                raw_x <- rand_normal(sigma = 0.3)
                closed_form(~stand_dist(raw_x))
              },
              lognormal = {
                raw_x <- rand_lognormal(sigma = 0.6)
                closed_form(~stand_dist(raw_x/3 - 1))
              },
              even_discrete = rand_uniform_d(k = 5, even = TRUE))
  
  mod <- heter_model(a = model_parameters$a[i],
                     b = model_parameters$b[i],
                     x = x)
  
  heter_dat <- map(1:5, ~mod$gen(model_parameters$n[i]))
  
  draw_single_plots(heter_dat, "heter/test/not_null", i * 5)
}

for (i in 1:nrow(model_parameters)) {
  x <- switch(as.character(model_parameters$x_dist[i]),
              uniform = rand_uniform(-1, 1),
              normal = {
                raw_x <- rand_normal(sigma = 0.3)
                closed_form(~stand_dist(raw_x))
              },
              lognormal = {
                raw_x <- rand_lognormal(sigma = 0.6)
                closed_form(~stand_dist(raw_x/3 - 1))
              },
              even_discrete = rand_uniform_d(k = 5, even = TRUE))
  
  mod <- heter_model(a = model_parameters$a[i],
                     b = 0,
                     x = x)
  
  heter_dat <- map(1:50, ~mod$gen(model_parameters$n[i]))
  
  draw_single_plots(heter_dat, "heter/train/null", i * 50)
}

for (i in 1:nrow(model_parameters)) {
  x <- switch(as.character(model_parameters$x_dist[i]),
              uniform = rand_uniform(-1, 1),
              normal = {
                raw_x <- rand_normal(sigma = 0.3)
                closed_form(~stand_dist(raw_x))
              },
              lognormal = {
                raw_x <- rand_lognormal(sigma = 0.6)
                closed_form(~stand_dist(raw_x/3 - 1))
              },
              even_discrete = rand_uniform_d(k = 5, even = TRUE))
  
  mod <- heter_model(a = model_parameters$a[i],
                     b = 0,
                     x = x)
  
  heter_dat <- map(1:5, ~mod$gen(model_parameters$n[i]))
  
  draw_single_plots(heter_dat, "heter/test/null", i * 5)
}
