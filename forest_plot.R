if (!require("metafor")) install.packages("metafor")
if (!require("readxl")) install.packages("readxl")
library(metafor)
library(readxl)

# ---  Set up the EPS file device ---
postscript("forest_plot_1200dpi_equivalent.eps",
           width = 8, height = 6, # Dimensions in inches
           horizontal = FALSE,
           onefile = FALSE,
           paper = "special")


# Load data
data <- read_excel("Data_Extracted.xlsx", sheet = "main")
study_labels <- c("San Diego", "Turku", "Iowa", "New Mexico", "OMEGA", "NatMEG")

# Set up plot layout
par(mfrow = c(3, 2), mar = c(4, 4, 2, 2))

# (A) Alpha peak amplitude
n_PD <- data$"PD ON med participants"
n_HC <- data$"HC participants"
AlphaAmp_avg_PD <- data$"PD alpha-ample"
AlphaAmp_SD_PD <- data$"PD alpha-ample SD"
AlphaAmp_avg_HC <- data$"HC alpha-ample"
AlphaAmp_SD_HC <- data$"HC alpha-ample SD"
AlphaAmp_effect_size <- escalc(measure = "SMD", m1i = AlphaAmp_avg_PD, sd1i = AlphaAmp_SD_PD, n1i = n_PD, m2i = AlphaAmp_avg_HC, sd2i = AlphaAmp_SD_HC, n2i = n_HC)
AlphaAmp_meta_analysis <- rma(yi = AlphaAmp_effect_size$yi, vi = AlphaAmp_effect_size$vi)
p_value_alpha_amp <- formatC(AlphaAmp_meta_analysis$pval, format = "e", digits = 1)
forest(AlphaAmp_meta_analysis, 
       slab = study_labels, main = "",
       xlab = "", xlim = c(-3.5, 3.5), 
       at = c(-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3))


mtext("(A) Alpha peak amplitude", side = 3, line = 0.25, adj = 0, cex = 1) 
text(2.2, -1.52, paste("p =", p_value_alpha_amp), adj = 0, cex = 0.7)


# (B) Alpha peak frequency
AlphaCF_avg_PD <- data$"PD alpha-CF"
AlphaCF_SD_PD <- data$"PD alpha-CF SD"
AlphaCF_avg_HC <- data$"HC alpha-CF"
AlphaCF_SD_HC <- data$"HC alpha-CF SD"
AlphaCF_effect_size <- escalc(measure = "SMD", m1i = AlphaCF_avg_PD, sd1i = AlphaCF_SD_PD, n1i = n_PD, m2i = AlphaCF_avg_HC, sd2i = AlphaCF_SD_HC, n2i = n_HC)
AlphaCF_meta_analysis <- rma(yi = AlphaCF_effect_size$yi, vi = AlphaCF_effect_size$vi)
p_value_alpha_cf <- formatC(AlphaCF_meta_analysis$pval, format = "e", digits = 2)
forest(AlphaCF_meta_analysis, slab = study_labels, main = "", xlab = "",
       xlim = c(-3.5, 3.5), 
       at = c(-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3))

mtext("(B) Alpha peak frequency", side = 3, line = 0.25, adj = 0, cex = 1) 

text(2.2, -1.52, paste("p =", p_value_alpha_cf), adj = 0, cex = 0.7)

# (C) Beta peak amplitude
BetaAmp_avg_PD <- data$"PD beta-ample"
BetaAmp_SD_PD <- data$"PD beta-ample SD"
BetaAmp_avg_HC <- data$"HC beta-ample"
BetaAmp_SD_HC <- data$"HC beta-ample SD"
BetaAmp_effect_size <- escalc(measure = "SMD", m1i = BetaAmp_avg_PD, sd1i = BetaAmp_SD_PD, n1i = n_PD, m2i = BetaAmp_avg_HC, sd2i = BetaAmp_SD_HC, n2i = n_HC)
BetaAmp_meta_analysis <- rma(yi = BetaAmp_effect_size$yi, vi = BetaAmp_effect_size$vi)
p_value_beta_amp <- formatC(BetaAmp_meta_analysis$pval, format = "e", digits = 2)
forest(BetaAmp_meta_analysis, slab = study_labels, main = "", xlab = "", 
       xlim = c(-3.5, 3.5), 
       at = c(-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3))
mtext("(C) Beta peak amplitude", side = 3, line = 0.25, adj = 0, cex = 1)
text(2.2, -1.52, paste("p =", p_value_beta_amp), adj = 0, cex = 0.7)

# (D) Beta peak frequency
BetaCF_avg_PD <- data$"PD beta-CF"
BetaCF_SD_PD <- data$"PD beta-CF SD"
BetaCF_avg_HC <- data$"HC beta-CF"
BetaCF_SD_HC <- data$"HC beta-CF SD"
BetaCF_effect_size <- escalc(measure = "SMD", m1i = BetaCF_avg_PD, sd1i = BetaCF_SD_PD, n1i = n_PD, m2i = BetaCF_avg_HC, sd2i = BetaCF_SD_HC, n2i = n_HC)
BetaCF_meta_analysis <- rma(yi = BetaCF_effect_size$yi, vi = BetaCF_effect_size$vi)
p_value_beta_cf <- formatC(BetaCF_meta_analysis$pval, format = "e", digits = 2)
forest(BetaCF_meta_analysis, slab = study_labels, main = "", xlab = "",
       xlim = c(-3.5, 3.5), 
       at = c(-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3))
mtext("(D) Beta peak frequency", side = 3, line = 0.25, adj = 0, cex = 1)
text(2.2, -1.52, paste("p =", p_value_beta_cf), adj = 0, cex = 0.7)

# (E) Exponent
exp_avg_PD <- data$"PD exp"
exp_SD_PD <- data$"PD exp SD"
exp_avg_HC <- data$"HC exp"
exp_SD_HC <- data$"HC exp SD"
Exp_effect_size <- escalc(measure = "SMD", m1i = exp_avg_PD, sd1i = exp_SD_PD, n1i = n_PD, m2i = exp_avg_HC, sd2i = exp_SD_HC, n2i = n_HC)
Exp_meta_analysis <- rma(yi = Exp_effect_size$yi, vi = Exp_effect_size$vi)
p_value_exp <- formatC(Exp_meta_analysis$pval, format = "e", digits = 2)
forest(Exp_meta_analysis, slab = study_labels, main = "", xlab = "", 
       xlim = c(-3.5, 3.5), 
       at = c(-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3))
mtext("(E) Exponent", side = 3, line = 0.25, adj = 0, cex = 1)
text(2.2, -1.52, paste("p =", p_value_exp), adj = 0, cex = 0.7)

# (F) Offset
offset_avg_PD <- data$"PD offset"
offset_SD_PD <- data$"PD offset SD"
offset_avg_HC <- data$"HC offset"
offset_SD_HC <- data$"HC offset SD"
offset_effect_size <- escalc(measure = "SMD", m1i = offset_avg_PD, sd1i = offset_SD_PD, n1i = n_PD, m2i = offset_avg_HC, sd2i = offset_SD_HC, n2i = n_HC)
offset_meta_analysis <- rma(yi = offset_effect_size$yi, vi = offset_effect_size$vi)
p_value_offset <- formatC(offset_meta_analysis$pval, format = "e", digits = 2)
forest(offset_meta_analysis, slab = study_labels, main = "", xlab = "", 
       xlim = c(-3.5, 3.5), 
       at = c(-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3))
mtext("(F) Offset", side = 3, line = 0.25, adj = 0, cex = 1)
text(2.2, -1.52, paste("p =", p_value_offset), adj = 0, cex = 0.7)

mtext("Standardized Mean Difference", side = 1, outer = TRUE, line = 2)


dev.off()