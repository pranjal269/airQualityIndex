# Statistical Analysis & EDA using R
# This script performs comprehensive statistical analysis on the ML-ready data
# Outputs: Correlation analysis, ANOVA tests, time series decomposition, visualizations

# Load required libraries
library(tidyverse)
library(corrplot)
library(ggplot2)
library(lubridate)
library(gridExtra)

# Set working directory to project root
# When running from Rscript, use command line args or current directory
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])

if (length(script_path) > 0) {
  # Running from Rscript with file path
  script_dir <- dirname(script_path)
  setwd(dirname(script_dir))
} else {
  # Running interactively or from current directory
  if (basename(getwd()) == "scripts") {
    setwd("..")
  }
}

# Create output directory for R analysis
dir.create("data/processed/r_analysis", showWarnings = FALSE)

cat("================================================================================\n")
cat("Starting Statistical Analysis & EDA (R)\n")
cat("================================================================================\n")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
cat("\n1. Loading ML-ready data...\n")
df <- read_csv("data/processed/ml_ready_data.csv", show_col_types = FALSE)
cat(sprintf("   Loaded %s records with %s features\n", format(nrow(df), big.mark=","), ncol(df)))

# Convert datetime
df$datetime_utc <- as.POSIXct(df$datetime_utc, format="%Y-%m-%d %H:%M:%S", tz="UTC")

# ============================================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================================
cat("\n2. Computing descriptive statistics...\n")

# Summary statistics for key pollutants and AQI
pollutants <- c("pm25", "pm10", "no2", "so2", "co", "o3", "AQI")
summary_stats <- df %>%
  select(all_of(pollutants)) %>%
  summary()

# Save summary statistics
sink("data/processed/r_analysis/descriptive_stats.txt")
cat("DESCRIPTIVE STATISTICS\n")
cat("================================================================================\n\n")
print(summary_stats)
sink()

cat("   Saved: data/processed/r_analysis/descriptive_stats.txt\n")

# ============================================================================
# 3. CORRELATION ANALYSIS
# ============================================================================
cat("\n3. Performing correlation analysis...\n")

# Correlation matrix for pollutants and AQI
cor_data <- df %>%
  select(all_of(pollutants)) %>%
  na.omit()

cor_matrix <- cor(cor_data)

# Save correlation matrix
png("data/processed/r_analysis/correlation_matrix.png", width=1200, height=1000, res=150)
corrplot(cor_matrix, 
         method="color", 
         type="upper",
         addCoef.col="black",
         tl.col="black",
         tl.srt=45,
         title="Correlation Matrix: Pollutants & AQI",
         mar=c(0,0,2,0),
         number.cex=0.8)
dev.off()

cat("   Saved: data/processed/r_analysis/correlation_matrix.png\n")

# ============================================================================
# 4. LOCATION-WISE ANALYSIS
# ============================================================================
cat("\n4. Analyzing location-wise patterns...\n")

# AQI by location
location_summary <- df %>%
  group_by(location_name) %>%
  summarise(
    count = n(),
    mean_aqi = mean(AQI, na.rm=TRUE),
    median_aqi = median(AQI, na.rm=TRUE),
    sd_aqi = sd(AQI, na.rm=TRUE),
    min_aqi = min(AQI, na.rm=TRUE),
    max_aqi = max(AQI, na.rm=TRUE)
  ) %>%
  arrange(desc(mean_aqi))

# Save location summary
write_csv(location_summary, "data/processed/r_analysis/location_summary.csv")
cat("   Saved: data/processed/r_analysis/location_summary.csv\n")

# Boxplot: AQI by location
png("data/processed/r_analysis/aqi_by_location.png", width=1400, height=800, res=150)
ggplot(df, aes(x=reorder(location_name, AQI, FUN=median), y=AQI, fill=location_name)) +
  geom_boxplot() +
  coord_flip() +
  labs(title="AQI Distribution by Location",
       x="Location",
       y="Air Quality Index (AQI)") +
  theme_minimal() +
  theme(legend.position="none",
        plot.title=element_text(hjust=0.5, size=16, face="bold"))
dev.off()

cat("   Saved: data/processed/r_analysis/aqi_by_location.png\n")

# ============================================================================
# 5. TEMPORAL PATTERNS
# ============================================================================
cat("\n5. Analyzing temporal patterns...\n")

# AQI by hour of day
hourly_pattern <- df %>%
  group_by(hour) %>%
  summarise(
    mean_aqi = mean(AQI, na.rm=TRUE),
    sd_aqi = sd(AQI, na.rm=TRUE)
  )

png("data/processed/r_analysis/aqi_hourly_pattern.png", width=1200, height=700, res=150)
ggplot(hourly_pattern, aes(x=hour, y=mean_aqi)) +
  geom_line(color="steelblue", size=1.2) +
  geom_ribbon(aes(ymin=mean_aqi-sd_aqi, ymax=mean_aqi+sd_aqi), alpha=0.2, fill="steelblue") +
  labs(title="Average AQI by Hour of Day",
       x="Hour of Day",
       y="Mean AQI") +
  scale_x_continuous(breaks=seq(0, 23, 2)) +
  theme_minimal() +
  theme(plot.title=element_text(hjust=0.5, size=16, face="bold"))
dev.off()

cat("   Saved: data/processed/r_analysis/aqi_hourly_pattern.png\n")

# AQI by day of week
weekly_pattern <- df %>%
  group_by(day_of_week) %>%
  summarise(
    mean_aqi = mean(AQI, na.rm=TRUE),
    sd_aqi = sd(AQI, na.rm=TRUE)
  )

png("data/processed/r_analysis/aqi_weekly_pattern.png", width=1200, height=700, res=150)
ggplot(weekly_pattern, aes(x=factor(day_of_week, labels=c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")), 
                            y=mean_aqi)) +
  geom_bar(stat="identity", fill="coral") +
  labs(title="Average AQI by Day of Week",
       x="Day of Week",
       y="Mean AQI") +
  theme_minimal() +
  theme(plot.title=element_text(hjust=0.5, size=16, face="bold"))
dev.off()

cat("   Saved: data/processed/r_analysis/aqi_weekly_pattern.png\n")

# ============================================================================
# 6. POLLUTANT DISTRIBUTIONS
# ============================================================================
cat("\n6. Analyzing pollutant distributions...\n")

# Create histograms for each pollutant
pollutant_plots <- list()

for (pollutant in c("pm25", "pm10", "no2", "so2", "co", "o3")) {
  p <- ggplot(df, aes_string(x=pollutant)) +
    geom_histogram(bins=50, fill="steelblue", alpha=0.7) +
    labs(title=toupper(pollutant), x=pollutant, y="Frequency") +
    theme_minimal() +
    theme(plot.title=element_text(hjust=0.5, face="bold"))
  
  pollutant_plots[[pollutant]] <- p
}

png("data/processed/r_analysis/pollutant_distributions.png", width=1600, height=1200, res=150)
grid.arrange(grobs=pollutant_plots, ncol=3)
dev.off()

cat("   Saved: data/processed/r_analysis/pollutant_distributions.png\n")

# ============================================================================
# 7. ANOVA TESTS
# ============================================================================
cat("\n7. Performing ANOVA tests...\n")

# ANOVA: AQI by location
anova_location <- aov(AQI ~ location_name, data=df)
anova_location_summary <- summary(anova_location)

# ANOVA: AQI by time of day
df$time_of_day <- cut(df$hour, 
                       breaks=c(0, 6, 12, 18, 24),
                       labels=c("Night", "Morning", "Afternoon", "Evening"),
                       include.lowest=TRUE)

anova_time <- aov(AQI ~ time_of_day, data=df)
anova_time_summary <- summary(anova_time)

# Save ANOVA results
sink("data/processed/r_analysis/anova_results.txt")
cat("ANOVA RESULTS\n")
cat("================================================================================\n\n")
cat("1. AQI by Location\n")
cat("------------------\n")
print(anova_location_summary)
cat("\n\n2. AQI by Time of Day\n")
cat("---------------------\n")
print(anova_time_summary)
sink()

cat("   Saved: data/processed/r_analysis/anova_results.txt\n")

# ============================================================================
# 8. TIME SERIES VISUALIZATION
# ============================================================================
cat("\n8. Creating time series visualizations...\n")

# Daily average AQI trend
daily_trend <- df %>%
  mutate(date = as.Date(datetime_utc)) %>%
  group_by(date, location_name) %>%
  summarise(mean_aqi = mean(AQI, na.rm=TRUE), .groups="drop")

png("data/processed/r_analysis/aqi_time_series.png", width=1600, height=800, res=150)
ggplot(daily_trend, aes(x=date, y=mean_aqi, color=location_name)) +
  geom_line(size=1) +
  labs(title="Daily Average AQI Trend by Location",
       x="Date",
       y="Mean AQI",
       color="Location") +
  theme_minimal() +
  theme(plot.title=element_text(hjust=0.5, size=16, face="bold"),
        legend.position="bottom")
dev.off()

cat("   Saved: data/processed/r_analysis/aqi_time_series.png\n")

# ============================================================================
# 9. AQI CATEGORY ANALYSIS
# ============================================================================
cat("\n9. Analyzing AQI categories...\n")

# AQI category distribution
category_dist <- df %>%
  count(AQI_Category) %>%
  mutate(percentage = n / sum(n) * 100) %>%
  arrange(desc(n))

png("data/processed/r_analysis/aqi_category_distribution.png", width=1200, height=800, res=150)
ggplot(category_dist, aes(x=reorder(AQI_Category, -n), y=n, fill=AQI_Category)) +
  geom_bar(stat="identity") +
  geom_text(aes(label=sprintf("%s\n(%.1f%%)", n, percentage)), vjust=-0.5) +
  labs(title="AQI Category Distribution",
       x="AQI Category",
       y="Count") +
  scale_fill_manual(values=c("Good"="green", "Moderate"="yellow", 
                              "Poor"="orange", "Very Poor"="red", "Severe"="darkred")) +
  theme_minimal() +
  theme(plot.title=element_text(hjust=0.5, size=16, face="bold"),
        legend.position="none")
dev.off()

cat("   Saved: data/processed/r_analysis/aqi_category_distribution.png\n")

# ============================================================================
# 10. SUMMARY REPORT
# ============================================================================
cat("\n10. Generating summary report...\n")

sink("data/processed/r_analysis/analysis_summary.txt")
cat("STATISTICAL ANALYSIS SUMMARY\n")
cat("================================================================================\n\n")
cat(sprintf("Analysis Date: %s\n", Sys.Date()))
cat(sprintf("Total Records: %s\n", format(nrow(df), big.mark=",")))
cat(sprintf("Date Range: %s to %s\n", min(df$datetime_utc), max(df$datetime_utc)))
cat(sprintf("Number of Locations: %s\n\n", n_distinct(df$location_name)))

cat("KEY FINDINGS:\n")
cat("-------------\n\n")

cat("1. Overall AQI Statistics:\n")
cat(sprintf("   - Mean AQI: %.1f\n", mean(df$AQI, na.rm=TRUE)))
cat(sprintf("   - Median AQI: %.1f\n", median(df$AQI, na.rm=TRUE)))
cat(sprintf("   - Std Dev: %.1f\n\n", sd(df$AQI, na.rm=TRUE)))

cat("2. Location with Highest Mean AQI:\n")
cat(sprintf("   - %s (Mean AQI: %.1f)\n\n", 
            location_summary$location_name[1], 
            location_summary$mean_aqi[1]))

cat("3. Most Common AQI Category:\n")
cat(sprintf("   - %s (%.1f%% of records)\n\n", 
            category_dist$AQI_Category[1], 
            category_dist$percentage[1]))

cat("4. Strongest Correlations with AQI:\n")
aqi_cors <- cor_matrix[,"AQI"]
aqi_cors <- sort(abs(aqi_cors[names(aqi_cors) != "AQI"]), decreasing=TRUE)
for (i in 1:min(3, length(aqi_cors))) {
  cat(sprintf("   - %s: %.3f\n", names(aqi_cors)[i], aqi_cors[i]))
}

cat("\n\nOUTPUT FILES GENERATED:\n")
cat("-----------------------\n")
cat("  1. descriptive_stats.txt - Summary statistics\n")
cat("  2. correlation_matrix.png - Correlation heatmap\n")
cat("  3. location_summary.csv - Location-wise statistics\n")
cat("  4. aqi_by_location.png - Boxplot by location\n")
cat("  5. aqi_hourly_pattern.png - Hourly AQI trends\n")
cat("  6. aqi_weekly_pattern.png - Weekly AQI patterns\n")
cat("  7. pollutant_distributions.png - Histograms\n")
cat("  8. anova_results.txt - ANOVA test results\n")
cat("  9. aqi_time_series.png - Time series plot\n")
cat("  10. aqi_category_distribution.png - Category breakdown\n")
cat("  11. analysis_summary.txt - This file\n")

sink()

cat("   Saved: data/processed/r_analysis/analysis_summary.txt\n")

cat("\n================================================================================\n")
cat("Statistical Analysis Complete!\n")
cat("================================================================================\n")
cat("All outputs saved to: data/processed/r_analysis/\n")
cat("\nNext step: Train ML models (6_train_models.py)\n")
