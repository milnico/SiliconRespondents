# 04_factor_recovery.R

suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(psych))

# Function to run factor recovery analysis for a given split
run_factor_recovery <- function(split_name) {
  cat("\n==========================================\n")
  cat("Running Factor Recovery for", split_name, "\n")
  cat("==========================================\n")
  
  # Load data
  film_data <- read.csv(paste0("predictions_film_", split_name, ".csv"))
  mlp_data <- read.csv(paste0("predictions_mlp_", split_name, ".csv"))
  
  # Pivot to wide format (Participants x Items)
  true_wide <- film_data %>%
    select(participant_id, item_id, true_resp) %>%
    pivot_wider(names_from = item_id, values_from = true_resp) %>%
    select(-participant_id)
  
  film_wide <- film_data %>%
    select(participant_id, item_id, pred_resp) %>%
    pivot_wider(names_from = item_id, values_from = pred_resp) %>%
    select(-participant_id)
    
  mlp_wide <- mlp_data %>%
    select(participant_id, item_id, pred_resp) %>%
    pivot_wider(names_from = item_id, values_from = pred_resp) %>%
    select(-participant_id)
    
  # Drop columns with zero variance to avoid FA errors
  true_vars <- apply(true_wide, 2, var, na.rm=TRUE)
  film_vars <- apply(film_wide, 2, var, na.rm=TRUE)
  mlp_vars <- apply(mlp_wide, 2, var, na.rm=TRUE)
  
  valid_items <- names(true_wide)[true_vars > 0 & film_vars > 0 & mlp_vars > 0]
  
  true_wide <- true_wide[, valid_items]
  film_wide <- film_wide[, valid_items]
  mlp_wide <- mlp_wide[, valid_items]
  
  cat("Valid items with variance > 0:", length(valid_items), "\n")
  
  # Factor Analysis
  cat("\nComputing 5-factor structure (minres, oblimin)...\n")
  fa_true <- fa(true_wide, nfactors = 5, fm = "minres", rotate = "oblimin", scores = "regression", warnings = FALSE)
  fa_film <- fa(film_wide, nfactors = 5, fm = "minres", rotate = "oblimin", scores = "regression", warnings = FALSE)
  fa_mlp <- fa(mlp_wide, nfactors = 5, fm = "minres", rotate = "oblimin", scores = "regression", warnings = FALSE)
  
  # Compute Congruence
  cat("\n--- Tucker's Congruence Coefficient ---\n")
  
  cong_film <- factor.congruence(fa_true$loadings, fa_film$loadings)
  cong_mlp <- factor.congruence(fa_true$loadings, fa_mlp$loadings)
  
  # Match factors based on max absolute congruence
  matched_film_cols <- apply(abs(cong_film), 1, which.max)
  film_sign <- sign(apply(cong_film, 1, function(x) x[which.max(abs(x))]))
  
  matched_mlp_cols <- apply(abs(cong_mlp), 1, which.max)
  mlp_sign <- sign(apply(cong_mlp, 1, function(x) x[which.max(abs(x))]))
  
  cat("Mean FiLM Congruence:", round(mean(apply(abs(cong_film), 1, max)), 3), "\n")
  cat("Mean MLP Congruence:", round(mean(apply(abs(cong_mlp), 1, max)), 3), "\n")
  
  # Factor Score Recovery
  cat("\n--- Factor Score Recovery (Corr true_factor vs synthetic_factor) ---\n")
  true_scores <- fa_true$scores
  film_scores <- fa_film$scores[, matched_film_cols] * matrix(film_sign, nrow=nrow(fa_film$scores), ncol=5, byrow=TRUE)
  mlp_scores <- fa_mlp$scores[, matched_mlp_cols] * matrix(mlp_sign, nrow=nrow(fa_mlp$scores), ncol=5, byrow=TRUE)
  
  film_score_corr <- diag(cor(true_scores, film_scores))
  mlp_score_corr <- diag(cor(true_scores, mlp_scores))
  
  print(round(film_score_corr, 3))
  cat("FiLM Mean Factor Score Recovery:", round(mean(film_score_corr), 3), "\n")
  
  print(round(mlp_score_corr, 3))
  cat("MLP Mean Factor Score Recovery:", round(mean(mlp_score_corr), 3), "\n")
  
  # Full Factor Correlation Matrix Error
  cat("\n--- Full Factor Correlation Matrix (Phi) Error ---\n")
  Phi_true <- fa_true$Phi
  
  Phi_film <- fa_film$Phi[matched_film_cols, matched_film_cols] * outer(film_sign, film_sign)
  Phi_mlp <- fa_mlp$Phi[matched_mlp_cols, matched_mlp_cols] * outer(mlp_sign, mlp_sign)
  
  frob_film <- norm(as.matrix(Phi_true - Phi_film), type="F")
  frob_mlp <- norm(as.matrix(Phi_true - Phi_mlp), type="F")
  
  cat("FiLM Phi Frobenius Norm Error:", round(frob_film, 4), "\n")
  cat("MLP Phi Frobenius Norm Error:", round(frob_mlp, 4), "\n")
}

run_factor_recovery("Split_C")
