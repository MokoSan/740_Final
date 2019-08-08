#######################
# Loading in the Data #
#######################

# Load in the data frame and get a perfunctory summary.
all_pulsar_data <- read.csv("./pulsar_stars.csv")
summary(all_pulsar_data)

# Check to see if any data cleaning is required.
any_na <- length(which(is.na(all_pulsar_data))); any_na

# No data cleaning required e.g. NA removals etc. time to move on to EDA.

#############################
# Exploratory Data Analysis #
#############################

# Integrated Profile
hist(Mean.of.the.integrated.profile)

# DM SNR Curve
hist(Mean.of.the.DM.SNR.curve)


summary(Mean.of.the.DM.SNR.curve)

non_pulsar <- pulsar_data[ pulsar_data$target_class == 0, ]
plot( x = non_pulsar$Mean.of.the.integrated.profile, 
      y = log( non_pulsar$Mean.of.the.DM.SNR.curve ))