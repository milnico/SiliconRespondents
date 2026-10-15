library(data.table)
library(lavaan)

cat("Reading codebook.txt...\n")
codebook_lines <- readLines("codebook.txt")
codebook_lines <- codebook_lines[codebook_lines != ""]

reverse_items <- c()
for (line in codebook_lines) {
  parts <- strsplit(line, "\t")[[1]]
  if (length(parts) > 1) {
    item_id <- trimws(parts[1])
    rest <- parts[2]
    if (grepl("true[[:space:]]*$", rest, ignore.case = TRUE)) {
      reverse_items <- c(reverse_items, item_id)
    }
  }
}

cat("Identified reverse items:\n")
print(reverse_items)

cat("Reading data.csv...\n")
data <- fread("data.csv")
num_rows_initial <- nrow(data)

# Remove rows with 0s if there are any (assume 1-5 scale)
# data[data == 0] <- NA
# data <- na.omit(data)
cat(sprintf("Loaded %d rows.\n", nrow(data)))

cat("Reverse scoring items...\n")
for (item in reverse_items) {
  if (item %in% names(data)) {
    data[[item]] <- 6 - data[[item]]
  }
}

cat("Building CFA model...\n")
model <- '
  O =~ O1 + O2 + O3 + O4 + O5 + O6 + O7 + O8 + O9 + O10
  C =~ C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 + C10
  E =~ E1 + E2 + E3 + E4 + E5 + E6 + E7 + E8 + E9 + E10
  A =~ A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10
  N =~ N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8 + N9 + N10
'

cat("Fitting CFA model (this may take a minute)...\n")
fit <- cfa(model, data = data, std.lv = TRUE, estimator = "ML")

cat("Extracting latent scores...\n")
pred <- lavPredict(fit)
pred_df <- as.data.frame(pred)

# Standardize latents
pred_df <- as.data.frame(scale(pred_df))
pred_df$participant_id <- paste0("P", sprintf("%05d", 1:nrow(pred_df)))

# Reorder columns
pred_df <- pred_df[, c("participant_id", "O", "C", "E", "A", "N")]

cat("Saving latent_scores.csv...\n")
fwrite(pred_df, "latent_scores.csv")
cat("Done!\n")
