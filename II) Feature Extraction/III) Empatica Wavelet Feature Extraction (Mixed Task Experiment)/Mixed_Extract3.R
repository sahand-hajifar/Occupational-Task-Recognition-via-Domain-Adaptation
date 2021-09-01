pacman::p_load(tidyverse, magrittr, doParallel,
               caret, caretEnsemble, AUC, caTools, DMwR, MLmetrics,
               rpart, nnet, #rpart and multinom packages
               randomForest, e1071, foreach, import, #parRF packages
               kernlab, # svmRadial package
               conflicted,ggfortify, readxl, DMwR, ROSE, pander,
               stringr, wavelets, entropy)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("./00_customFunctions.R") # custom functions

subject_numbers = c(4,6,8,9,10,12,15,16,17,20)
freq = 32

Subject_Data <- sprintf("./Data_Maxim/ACC%s.csv",subject_numbers) %>%
  lapply(read.csv) %>%
  lapply(function(x){x = x[-1,]; return(x)}) #remove first element which represents frequency

#This extracts the start time of each subject based on unix time
start_time = lapply(Subject_Data, function(x){temp = colnames(x)[1];return(temp)}) %>%
  lapply(function(x){temp = as.numeric(sub('.', '', x));return(temp)}) %>%
  unlist

Subject_Data = lapply(Subject_Data, function(x) {colnames(x) = c('ax','ay','az'); x})

# Filtering
# n = 4
# fc = 10
# Fs = 32
# Wn = (2/Fs)*fc
# but_filt <- signal::butter(n=n, W=Wn, type="low")

# Subject_Data = lapply(Subject_Data, function(x){
#   temp = apply(x, 2, signal::filter,filt = but_filt) ; temp})

subject_annotation = read_excel_allsheets("./Data_Maxim/mixed.xlsx")

to_be_removed = !apply(matrix(names(subject_annotation)), 1, function(x){ ### removes the subjects that are not in 'subject_numbers'
  temp = x %in% (sprintf("M%s",subject_numbers)); temp
})

subject_annotation[to_be_removed] = NULL

indices_list = vector(mode = "list", length = length(subject_annotation))
for (i in 1:length(indices_list)){
  indices_list[[i]] = freq * apply(matrix(subject_annotation[[i]][,3]), 1, time_diff, start_time = start_time[i])
}

Names = apply(matrix(subject_annotation[[1]][,2]), 1, function(x){name = sub("_.*", "",x);return(name)})

###

sub_range = 1:length(subject_numbers)

for (i in sub_range)
{
  Raw_Data<-ts(data = Subject_Data[[i]], start = 1, end = dim(Subject_Data[[i]])[1], frequency = 1, names =c("ax","ay","az") )
  vlines  = data.frame(xint = indices_list[[i]], grp = rep(c("start","end"),length(indices_list[[i]])/2))
  
  Plot = ggplot2::autoplot(Raw_Data)+theme_bw()+ ggtitle(paste0("Subject ",toString(subject_numbers[i])))+ theme(plot.title = element_text(hjust = 0.5))
  
  Plot = Plot + geom_vline(data = vlines,aes(xintercept = xint,colour = grp))
  
  print(Plot)
}

###

Names = Names[seq(2,length(Names),2)]
Names2 = paste0(Names, str_pad(1:length(Names), 3, pad = "0"))
feature_list = vector(mode = "list", length = length(subject_annotation))


for (num_sub in 1:length(Subject_Data)){
  
  indices_list2 = vector(mode = "list", length = length(Names))
  for (i in 1:length(indices_list2))
  {
    tryCatch({
      indices_list2[[i]] = indices_list[[num_sub]][2*i-1]:indices_list[[num_sub]][2*i]
    }, error = function(e) {})
  }
  
  df = data.frame(acc= Subject_Data[[num_sub]]*9.81/64, label = NA)
  for (i in 1:length(indices_list2)) { 
    rows = indices_list2[[i]]
    df[rows, 'label'] = Names2[i]
  }
  
  df %<>% na.omit()
  df$label = as.factor(df$label)
  numVars = 128 # we can change that
  numLevel = 5
  
  df = split(df, df$label)
  df_names = names(df)
  list_names = str_sub(df_names, 1,nchar(df_names)-3)
  
  for (ii in 1:length(df))
  {
    if (list_names[ii] == 'sit'|list_names[ii] == 'stand'|list_names[ii] == 'walk')
    {
      df[[ii]] = tail(df[[ii]], -numVars)
    }
    
  }
  
  df = do.call(rbind, df)
  
  feature_X = featureX(df)
  feature_Y = featureY(df)
  feature_Z = featureZ(df)
  
  feature = cbind(feature_X[,1:(dim(feature_X)[2]-1)],
                  feature_Y[,2:(dim(feature_Y)[2]-1)],
                  feature_Z[,2:(dim(feature_Z)[2])])
  
  feature$label = substr(feature$label,1,nchar(as.character(feature$label))-3)
  
  colnames(feature) = c('label', paste0("f", seq(1, 3*((numLevel+1)*12))), 'id')
  feature_list[[num_sub]] = feature
}

save(feature_list, file = "feature_list.Rdata")
