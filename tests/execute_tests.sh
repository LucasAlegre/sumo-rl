# Colour Coding
NC='\033[0m' 
RED='\033[1;31m' 

for f in *.py
do
    printf "${RED} Executing test: $f ${NC} \n"
    python3 "$f"
 done