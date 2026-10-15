# 04_simulator_factor_recovery.R
args <- commandArgs(trailingOnly = TRUE)
pred_file <- args[1]

suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(psych))

data <- read.csv(pred_file)

true_wide <- data %>%
  select(participant_id, item_id, true_resp) %>%
  pivot_wider(names_from = item_id, values_from = true_resp) %>%
  select(-participant_id)

pred_wide <- data %>%
  select(participant_id, item_id, pred_resp) %>%
  pivot_wider(names_from = item_id, values_from = pred_resp) %>%
  select(-participant_id)

true_vars <- apply(true_wide, 2, var, na.rm=TRUE)
pred_vars <- apply(pred_wide, 2, var, na.rm=TRUE)

valid_items <- names(true_wide)[true_vars > 0 & pred_vars > 0]

true_wide <- true_wide[, valid_items]
pred_wide <- pred_wide[, valid_items]

if (length(valid_items) < 5) {
   cat("Not enough valid items to compute 5-factor structure.\n")
   quit(save="no", status=0)
}

fa_true <- tryCatch(fa(true_wide, nfactors = min(5, length(valid_items)-1), fm = "minres", rotate = "oblimin", warnings = FALSE), error=function(e) NULL)
fa_pred <- tryCatch(fa(pred_wide, nfactors = min(5, length(valid_items)-1), fm = "minres", rotate = "oblimin", warnings = FALSE), error=function(e) NULL)

if (!is.null(fa_true) && !is.null(fa_pred)) {
    cong <- factor.congruence(fa_true$loadings, fa_pred$loadings)
    matched_cols <- apply(abs(cong), 1, which.max)
    pred_sign <- sign(apply(cong, 1, function(x) x[which.max(abs(x))]))
    
    mean_cong <- mean(apply(abs(cong), 1, max))
    cat("Factor Loading Congruence: ", round(mean_cong, 4), "\n")
    
    Phi_true <- fa_true$Phi
    Phi_pred <- fa_pred$Phi[matched_cols, matched_cols] * outer(pred_sign, pred_sign)
    frob_phi <- norm(as.matrix(Phi_true - Phi_pred), type="F")
    cat("Factor Correlation Matrix Error: ", round(frob_phi, 4), "\n")
} else {
    cat("Factor Analysis failed to converge.\n")
}
