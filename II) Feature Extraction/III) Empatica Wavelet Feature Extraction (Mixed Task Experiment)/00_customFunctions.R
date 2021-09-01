splitFun = function(x, numVars){
  temp = split(x, ceiling(seq_along(x)/numVars))
  temp %<>% .[-length(temp)] %>%
    lapply(wExtract)
  return(temp)
}

time_diff  = function(input , start_time)
{
  start = substring(as.POSIXct(start_time, origin="1970-01-01"),12,19)
  start_hour = as.numeric(substring(start,1,2))
  start_min = as.numeric(substring(start,4,5))
  start_sec = as.numeric(substring(start,7,8))
  
  current_hour = as.numeric(substring(input,10,11))
  current_min = as.numeric(substring(input,13,14))
  current_sec = as.numeric(substring(input,16,17))
  
  
  
  difference = (current_hour*3600 + current_min*60 + current_sec ) -
    (start_hour*3600 + start_min*60 + start_sec )
  return(difference)
}

featureX = function(df)
{
  df = df[,-c(2,3)]
  df %<>% 
    group_by(label)%>% # so we can apply fft by group
    summarize(fft = list(splitFun(acc.ax, numVars)))%>% # fft application with Mod
    unnest_longer(fft)%>% # unlisting such that every numVar period is one row
    unnest_wider(fft)
}

featureY = function(df)
{
  df = df[,-c(1,3)]
  df %<>% 
    group_by(label)%>% # so we can apply fft by group
    summarize(fft = list(splitFun(acc.ay, numVars))) %>% # fft application with Mod
    unnest_longer(fft) %>% # unlisting such that every numVar period is one row
    unnest_wider(fft)
}

featureZ = function(df)
{
  df = df[,-c(1,2)]
  df %<>% 
    group_by(label)%>% # so we can apply fft by group
    summarize(fft = list(splitFun(acc.az, numVars))) %>% # fft application with Mod
    unnest_longer(fft) %>% # unlisting such that every numVar period is one row
    unnest_wider(fft)
}

read_excel_allsheets <- function(filename, tibble = FALSE) {
  # I prefer straight data.frames
  # but if you like tidyverse tibbles (the default with read_excel)
  # then just pass tibble = TRUE
  sheets <- readxl::excel_sheets(filename)
  x <- lapply(sheets, function(X) readxl::read_excel(filename, sheet = X))
  if(!tibble) x <- lapply(x, as.data.frame)
  names(x) <- sheets
  x
}

wExtract = function(sig){
  wt = dwt(sig, filter="d4", n.levels=5, boundary="reflection", fast=FALSE)
  coefs = c(wt@W, wt@V[length(wt@V)])
  features = lapply(coefs, function(x){
    temp = c(getStats(x), getCrossings(x), entropy(table(x)))
  })
  features = unlist(features)
  return(features)
}

getStats = function(x){
  Mean = mean(x)
  Rms = sqrt(mean(x^2))
  Mad = mean(abs(x-mean(x)))
  Std = sd(x)
  Min = min(x)
  Max = max(x)
  Med = quantile(x, 0.5)
  Perc25 = quantile(x, 0.25)
  Perc75 = quantile(x, 0.75)
  stats = c(Mean, Rms, Mad, Std, Min, Max, Med, Perc25, Perc75)
  return (stats)
}

getCrossings = function(x){
  zeroCross = sum((diff(sign(x))) !=  0)
  centeredSig = x - mean(x)
  meanCross = sum((diff(sign(centeredSig))) !=  0)
  return (c(zeroCross, meanCross))
}
