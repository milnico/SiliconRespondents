#!/usr/bin/env Rscript
# Compare the psychometric structure of a silicon sample against humans.
#
# Usage: Rscript 05_simulator_psychometrics.R human_norm.csv silicon_norm.csv
# Each input is long format with columns: participant_id,item_id,factor,reverse,resp
# (the Python driver 05_simulator_psychometrics.py produces these).
#
# Reports, on the items common to both files (ideally the held-out items):
#   * Tucker's congruence between oblimin factor loadings (>= 0.90 = replication)
#   * Inter-factor correlation matrix Phi (Frobenius error)
#   * Cronbach's alpha + corrected item-total (r.drop) per scale
#   * Correlation-matrix eigenvalue profile

suppressPackageStartupMessages({
  library(data.table)
  library(psych)
  library(GPArotation)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript 05_simulator_psychometrics.R human.csv silicon.csv")
human_file <- args[1]
silicon_file <- args[2]
n_factors <- 5

h <- fread(human_file)
s <- fread(silicon_file)

common <- sort(intersect(unique(h$item_id), unique(s$item_id)))
cat(sprintf("Common items: %d  ->  %s\n", length(common), paste(common, collapse = ", ")))

# item -> factor / reverse map (from the human file, authoritative)
meta <- unique(h[item_id %in% common, .(item_id, factor, reverse)])
setkey(meta, item_id)

to_wide <- function(dt) {
  w <- dcast(dt[item_id %in% common], participant_id ~ item_id, value.var = "resp",
             fun.aggregate = mean)
  w[, participant_id := NULL]
  as.data.frame(w)[, common, drop = FALSE]
}

reverse_score <- function(w) {
  for (it in colnames(w)) if (isTRUE(meta[it, reverse])) w[[it]] <- 6 - w[[it]]
  w
}

hw <- reverse_score(to_wide(h))
sw <- reverse_score(to_wide(s))

# Drop zero-variance items (e.g. an unconditioned baseline that gives the same
# answer to an item for every respondent) so correlation/FA stay well-defined.
nzv <- intersect(colnames(hw)[apply(hw, 2, sd, na.rm = TRUE) > 0],
                 colnames(sw)[apply(sw, 2, sd, na.rm = TRUE) > 0])
if (length(nzv) < length(common))
  cat(sprintf("[note] dropped %d zero-variance item(s): %s\n",
              length(common) - length(nzv), paste(setdiff(common, nzv), collapse = ", ")))
common <- nzv
hw <- hw[, common, drop = FALSE]
sw <- sw[, common, drop = FALSE]

# ---- Eigenvalues ----------------------------------------------------------
eig <- function(w) sort(eigen(cor(w, use = "pairwise.complete.obs"))$values, decreasing = TRUE)
cat("\nEigenvalues (top 6):\n")
cat("  human:  ", round(head(eig(hw), 6), 2), "\n")
cat("  silicon:", round(head(eig(sw), 6), 2), "\n")

# ---- Cronbach alpha + corrected item-total per scale ----------------------
cat("\nPer-scale Cronbach alpha and mean corrected item-total (r.drop):\n")
itc_h <- c(); itc_s <- c()
for (fac in c("O", "C", "E", "A", "N")) {
  cols <- meta[factor == fac & item_id %in% common, item_id]
  if (length(cols) < 2) next
  ah <- suppressWarnings(psych::alpha(hw[, cols], warnings = FALSE, check.keys = FALSE))
  as_ <- suppressWarnings(psych::alpha(sw[, cols], warnings = FALSE, check.keys = FALSE))
  itc_h <- c(itc_h, setNames(ah$item.stats$r.drop, cols))
  itc_s <- c(itc_s, setNames(as_$item.stats$r.drop, cols))
  cat(sprintf("  %s: alpha human=%.3f silicon=%.3f | mean r.drop human=%.3f silicon=%.3f (k=%d)\n",
              fac, ah$total$raw_alpha, as_$total$raw_alpha,
              mean(ah$item.stats$r.drop), mean(as_$item.stats$r.drop), length(cols)))
}
itc_s <- itc_s[names(itc_h)]
cat(sprintf("  item-total profile corr (human vs silicon) = %.3f  |  MAE = %.3f\n",
            cor(itc_h, itc_s), mean(abs(itc_h - itc_s))))
cat(sprintf("  mean corrected item-total (over items): human=%.3f silicon=%.3f\n",
            mean(itc_h), mean(itc_s)))

# ---- Factor congruence + Phi ---------------------------------------------
if (length(common) < 2 * n_factors) {
  cat(sprintf("\n[skip] %d items < %d needed for a %d-factor analysis.\n",
              length(common), 2 * n_factors, n_factors))
  quit(save = "no")
}
if (length(common) < 15)
  cat(sprintf("\n[warn] only %d items / %d factors -- FA is marginal.\n", length(common), n_factors))

fa_h <- fa(hw, nfactors = n_factors, rotate = "oblimin", fm = "minres", warnings = FALSE)
fa_s <- fa(sw, nfactors = n_factors, rotate = "oblimin", fm = "minres", warnings = FALSE)

C <- factor.congruence(fa_h$loadings, fa_s$loadings)  # human x silicon
# Greedy one-to-one matching on |congruence|.
A <- abs(C); ord <- integer(n_factors); sgn <- numeric(n_factors); val <- numeric(n_factors)
M <- A
for (i in 1:n_factors) {
  idx <- which(M == max(M), arr.ind = TRUE)[1, ]
  ord[idx[1]] <- idx[2]; val[idx[1]] <- C[idx[1], idx[2]]
  sgn[idx[1]] <- sign(C[idx[1], idx[2]])
  M[idx[1], ] <- -Inf; M[, idx[2]] <- -Inf
}
cat("\nTucker factor congruence (human factor -> best silicon factor):\n")
cat("  per factor:", round(abs(val), 3), "\n")
cat(sprintf("  MEAN congruence = %.3f   (replication target >= 0.90)\n", mean(abs(val))))

Phi_h <- fa_h$Phi
D <- diag(sgn)
Phi_s <- D %*% fa_s$Phi[ord, ord] %*% D
cat(sprintf("  Phi (inter-factor corr) Frobenius error = %.3f\n",
            norm(as.matrix(Phi_h - Phi_s), type = "F")))
