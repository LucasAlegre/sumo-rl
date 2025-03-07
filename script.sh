function process_file() {
  BASE=$1
  FILE=$2
  CONF=$3
  mv outputs/$FILE outputs/$BASE
  rm -rf outputs/$BASE
  python -m main -s $BASE $CONF
  python -m sumo_rl.util.plot -s $BASE
  mv outputs/$BASE outputs/$FILE
}

process_file celoria celoria-fx -f
process_file celoria celoria-rl-sj
process_file celoria celoria-rl-ag -a
