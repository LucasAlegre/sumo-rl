function process_file() {
  BASE=$1
  FILE=$2
  mv outputs/$FILE outputs/$BASE
  rm -rf outputs/$BASE/plots
  python -m sumo_rl.util.plot -s $BASE
  mv outputs/$BASE outputs/$FILE
}

process_file celoria celoria-fx
process_file celoria celoria-rl-ag
process_file celoria celoria-rl-sj
