# Recode agreement scale to numeric
RecodeAgreement <- function(x, type="direct") {
  if(x == "" | is.na(x) | !type %in% c("direct", "reversed")) return(NA)
  if(type=="reversed") {
    if(x == "Strongly disagree") return(5)
    if(x == "Somewhat disagree") return(4)
    if(x == "Neither agree nor disagree") return(3)
    if(x == "Somewhat agree") return(2)
    if(x == "Strongly agree") return(1)
  }
  if(type=="direct"){
    if(x == "Strongly disagree") return(1)
    if(x == "Somewhat disagree") return(2)
    if(x == "Neither agree nor disagree") return(3)
    if(x == "Somewhat agree") return(4)
    if(x == "Strongly agree") return(5)
  }
}

