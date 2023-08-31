from ray.rllib.examples import coin_game_env as coin
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--tf", action="store_true")
parser.add_argument("--stop-iters", type=int, default=2000)

if __name__ == "__main__":
    args = parser.parse_args()
    debug_mode = False
    use_asymmetric_env = False
    coin.main(debug_mode, args.stop_iters, args.tf, use_asymmetric_env)
