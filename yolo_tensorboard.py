import argparse
import subprocess


def build_parser():
    parser = argparse.ArgumentParser(description="Launch TensorBoard for YOLO training logs.")
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="6006")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    command = ["tensorboard", "--host", args.host, "--logdir", args.logdir, "--port", args.port]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
