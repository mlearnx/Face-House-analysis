## typical_difference_demo

# Code to reproduce the figures in my blog post on the shift function
# Copyright (C) 2016 Guillaume Rousselet - University of Glasgow

## ============================
## dependencies
# The text files should be in working directory of this script,
# or you can specify the path in each `source()` command:
setwd("/home/claire/SCRIPTS/R/blog_rousselet/typical_difference") # set working directory to "figure1" folder
source("rgar_stats.txt")
# packages:
library("ggplot2")
library("cowplot")
library("tidyr")
## ============================

# --------------------------------
# Dependent groups 
# --------------------------------

# load data
data<-read.csv("/home/claire/DATA/Data_Face_House_new_proc/power_density_all_freq.csv", header = TRUE, sep = ",", dec = ".")

data<-data[which(data$Freq=='theta'),]

# ----------------------------------------------------
# stripchart + linked observations -----------------
# mytitle = paste("Paired observations", freq_data$Freq[1])
# p <- ggplot(freq_data, aes(x=Modality, y=Median, fill=Modality, group=Subject)) +
#   theme_bw() +
#   geom_line(size=1, alpha=1) +
#   geom_point(colour = "black", size = 4, stroke = 1) +
#   geom_text(aes(label=Subject), hjust=0, vjust=0)+
#   #scale_shape_manual(values=c(22,21)) +
#   labs(title=mytitle  ) 
#   #scale_y_continuous(limits=c(0, 20),breaks=seq(0,20,5))
# # p
# linkedstrip <- p +
#   theme(strip.text.x = element_text(size = 20, colour = "white"),
#         strip.background = element_rect(colour="darkgrey", fill="darkgrey"))
# linkedstrip

# ----------------------------------------------------
# scatterplot of paired observations 
#----------------------------------------------------
#mytitle = paste("Median Power spectrum on Occ elec Stim-Imag")

perception <- data$Median[data$Modality=="stim"]
imagery <- data$Median[data$Modality=="imag"]
participant <- data$Subject[data$Modality=="imag"]
group <- data$Freq[data$Modality=="stim"]
#group <- alpha_data$Freq[data$condition=="condition2"]
rdata <- data.frame(participant, group, perception, imagery)
scatterdiff <- ggplot(rdata, aes(x=perception,y=imagery,  group=group,fill=group,colour=group,shape=group)) + 
  geom_abline(intercept = 0) +
  geom_point(size=4,stroke=1) +
  geom_text(aes(label=participant), hjust=0, vjust=0)+
  theme_bw() +
  scale_shape_manual(values=c(22,21,20,23, 24, 17, 18 )) +

  scale_fill_manual(values = c("#56B4E9", "#009E73")) +
  scale_colour_manual(values = c("#56B4E9", "#009E73")) +
  theme(axis.text.x = element_text(colour="grey20",size=16),
        axis.text.y = element_text(colour="grey20",size=16),  
        axis.title.x = element_text(colour="grey20",size=18),
        axis.title.y = element_text(colour="grey20",size=18),
        legend.title = element_blank(),
        legend.position = c(0.8, 0.2),
        #plot.margin = unit(c(150,100,5.5,5.5), "pt"),
        #plot.margin = unit(c(5.5,5.5,5.5,5.5), "pt"),
        legend.text = element_text(colour="grey20",size=16),
        plot.title = element_text(colour="grey20",size=20))
  #labs(title=mytitle) 
  #scale_x_continuous(limits=c(6, 16),breaks=seq(6,16,2)) +
  #scale_y_continuous(limits=c(6, 19),breaks=seq(6,19,2))
scatterdiff


# ----------------------------------------------------
# scatterplot of paired observations 
# Stim type
#-----------------------------------------------------


# load data
data<-read.csv("/home/claire/DATA/Data_Face_House_new_proc/power_density_all_freq_stim_type.csv", header = TRUE, sep = ",", dec = ".")

mytitle = paste("Median Power spectrum Imag face vs Imag house")

face <- data$Median[data$Modality=="imag/face"]
house <- data$Median[data$Modality=="imag/house"]
participant <- data$Subject[data$Modality=="stim/face"]
group <- data$Freq[data$Modality=="stim/face"]
#group <- alpha_data$Freq[data$condition=="condition2"]
rdata <- data.frame(participant, group, face, house)
scatterdiff <- ggplot(rdata, aes(x=face,y=house,  group=group,fill=group,colour=group,shape=group)) + 
  geom_abline(intercept = 0) +
  geom_point(size=4,stroke=1) +
  geom_text(aes(label=participant), hjust=0, vjust=0)+
  theme_bw() +
  scale_shape_manual(values=c(22,21,20,23, 24, 17, 18 )) +
  
  scale_fill_manual(values = c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00")) +
  scale_colour_manual(values = c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00")) +
  theme(axis.text.x = element_text(colour="grey20",size=16),
        axis.text.y = element_text(colour="grey20",size=16),  
        axis.title.x = element_text(colour="grey20",size=18),
        axis.title.y = element_text(colour="grey20",size=18),
        legend.title = element_blank(),
        legend.position = c(0.8, 0.2),
        #plot.margin = unit(c(150,100,5.5,5.5), "pt"),
        #plot.margin = unit(c(5.5,5.5,5.5,5.5), "pt"),
        legend.text = element_text(colour="grey20",size=16),
        plot.title = element_text(colour="grey20",size=20))+
  labs(title=mytitle) 
#scale_x_continuous(limits=c(6, 16),breaks=seq(6,16,2)) +
#scale_y_continuous(limits=c(6, 19),breaks=seq(6,19,2))
scatterdiff


# ----------------------------------------------------
# scatterplot of paired observations 
# Stim type
#-----------------------------------------------------


# load data
data<-read.csv("/home/claire/DATA/Data_Face_House_new_proc/power_density_all_freq_stim_type.csv", header = TRUE, sep = ",", dec = ".")

data<-data[which(data$Freq=='delta'),]

face <- data$Median[data$Modality=="imag/face"]
house <- data$Median[data$Modality=="imag/house"]
participant <- data$Subject[data$Modality=="stim/face"]
group <- data$Freq[data$Modality=="stim/face"]
#group <- alpha_data$Freq[data$condition=="condition2"]
rdata <- data.frame(participant, group, face, house)
scatterdiff <- ggplot(rdata, aes(x=face,y=house,  group=group,fill=group,colour=group,shape=group)) + 
  geom_abline(intercept = 0) +
  geom_point(size=4,stroke=1) +
  geom_text(aes(label=participant), hjust=0, vjust=0)+
  theme_bw() +
  scale_shape_manual(values=c(22,21,20,23, 24, 17, 18 )) +
  
  scale_fill_manual(values = c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00")) +
  scale_colour_manual(values = c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00")) +
  theme(axis.text.x = element_text(colour="grey20",size=16),
        axis.text.y = element_text(colour="grey20",size=16),  
        axis.title.x = element_text(colour="grey20",size=18),
        axis.title.y = element_text(colour="grey20",size=18),
        legend.title = element_blank(),
        legend.position = c(0.8, 0.2),
        #plot.margin = unit(c(150,100,5.5,5.5), "pt"),
        #plot.margin = unit(c(5.5,5.5,5.5,5.5), "pt"),
        legend.text = element_text(colour="grey20",size=16))
#scale_x_continuous(limits=c(6, 16),breaks=seq(6,16,2)) +
#scale_y_continuous(limits=c(6, 19),breaks=seq(6,19,2))
scatterdiff

