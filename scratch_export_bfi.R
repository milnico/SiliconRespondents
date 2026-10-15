library(psych)
data(bfi)
data(bfi.dictionary)

write.csv(bfi, "bfi_responses.csv", row.names=TRUE)
write.csv(bfi.dictionary, "bfi_dictionary.csv", row.names=TRUE)
